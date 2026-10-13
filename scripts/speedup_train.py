"""GT + v4q-832 heatmap/keypoint/query-output distillation; train/VAL only."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_fretnet import losses, validate

from gro.fretnet.data import PoseDataset
from gro.fretnet.model import build, load, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--init", type=Path, help="Continue a VAL-selected student; teacher remains v4q-832")
    ap.add_argument("--teacher", type=Path, default=ROOT / "runs/fretnet/v4q_r50_832/last.pt")
    a = ap.parse_args()
    if a.finetune and a.init:
        ap.error("--init is only for student continuation")
    torch.set_num_threads(4)
    torch.manual_seed(832)
    torch.backends.cudnn.benchmark = True
    out = ROOT / "runs/fretnet" / a.name
    out.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/datasets/scale-v1aug"
    train = PoseDataset(data, "train", size=832, train=True)
    val = PoseDataset(data, "val", size=a.imgsz, train=False)
    cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
    forbidden = [c["id"] for c in cfg["clips"]] + ["09-06", "09_06"]
    assert len(train) == 5798 and len(val) == 1498
    assert not any(s in p.name for ds in (train, val) for p in ds.images for s in forbidden)
    provenance = {k: str(v) for k, v in vars(a).items()}
    provenance.update(
        teacher_sha256=hashlib.sha256(a.teacher.read_bytes()).hexdigest(),
        train_sha256=hashlib.sha256("\n".join(p.name for p in train.images).encode()).hexdigest(),
        val_sha256=hashlib.sha256("\n".join(p.name for p in val.images).encode()).hexdigest(),
        selection="minimum VAL mean keypoint error",
        loss="GT CE + 0.5 L1 + BCE; teacher 0.5 spatial CE + 0.1 keypoint L1 + 0.2 query MSE + 0.2 confidence BCE",
    )
    if a.finetune:
        provenance["loss"] = "GT CE + 0.5 L1 + BCE; no distillation term"
    if a.init:
        provenance["init_sha256"] = hashlib.sha256(a.init.read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(provenance, indent=2))
    tl = DataLoader(train, a.batch, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val, 16, num_workers=4, pin_memory=True, persistent_workers=True)
    teacher = load(a.teacher, "cuda").requires_grad_(False)
    if a.finetune:
        net = load(a.teacher, "cuda")
    elif a.init:
        net = load(a.init, "cuda")
        assert net.arch == a.arch
    else:
        net = build(a.arch, "query", pretrained=True, imgsz=a.imgsz).cuda()
    net.imgsz = a.imgsz
    # Transfer compatible neck/query heads; channel-changing backbone projections stay freshly initialized.
    if not a.finetune and a.init is None:
        state = net.state_dict()
        copied = {
            k: v
            for k, v in teacher.state_dict().items()
            if not k.startswith(("stem.", "layers.")) and k in state and state[k].shape == v.shape
        }
        net.load_state_dict(copied, strict=False)
        provenance["transferred_tensors"] = len(copied)
    tq, sq = {}, {}
    teacher.glob.kernel.register_forward_hook(lambda m, i, o: tq.update(kernel=o))
    net.glob.kernel.register_forward_hook(lambda m, i, o: sq.update(kernel=o))
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    best = float("inf")
    log = (out / "log.jsonl").open("w")
    for epoch in range(a.epochs):
        start = time.perf_counter()
        total = 0.0
        for group in opt.param_groups:
            group["lr"] = 2e-4 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / a.epochs)))
        for x, k in tl:
            x = x.cuda(non_blocking=True).float() / 255
            k = k.cuda(non_blocking=True)
            k[..., :2] = (k[..., :2] + 0.5) * (a.imgsz / 832) - 0.5
            small = F.interpolate(x, size=(a.imgsz, a.imgsz), mode="bilinear", align_corners=False)
            if not a.finetune:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    th, tc = teacher(x)
                    target = F.interpolate(
                        th.float(), size=(a.imgsz // 4, a.imgsz // 4), mode="bilinear", align_corners=False
                    )
                    target = target.flatten(2).softmax(-1)
                    txy = (teacher.locate(th.float()) + 0.5) * (a.imgsz / 832) - 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sh, sc = net(small)
            ls = losses(net, sh, sc, k, 1.0)
            loss = ls["ce"] + 0.5 * ls["l1"] + ls["bce"]
            if not a.finetune:
                confidence = tc.float().sigmoid().detach()
                ce = -(target * sh.float().flatten(2).log_softmax(-1)).sum(-1)
                xy = (net.locate(sh.float()) - txy).abs().sum(-1) / 4
                loss = loss + 0.5 * (ce * confidence).mean() + 0.1 * (xy * confidence).mean()
                loss = loss + 0.2 * F.mse_loss(sq["kernel"].float(), tq["kernel"].float())
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(sc.float(), confidence)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            total += float(loss.detach())
        rec = dict(epoch=epoch + 1, loss=total / len(tl), **validate(net, vl, "cuda"))
        rec["seconds"] = time.perf_counter() - start
        if rec["err_mean_px"] < best:
            best = rec["err_mean_px"]
            save(net, out / "best.pt", **rec)
        save(net, out / "last.pt", **rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    (out / "protocol.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
