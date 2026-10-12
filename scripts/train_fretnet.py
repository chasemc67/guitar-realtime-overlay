"""Train FretNet (model v3, permissive licence; no Ultralytics) on the YOLO-pose export of scale-v1 / scale-v1aug.

    .venv/bin/python scripts/train_fretnet.py --name v3_r50 --data data/datasets/scale-v1aug --arch resnet50

Loss per labelled keypoint: cross-entropy between softmax(heatmap) and a Gaussian target (sigma in heatmap cells),
L1 of the windowed soft-argmax location (heatmap cells), and BCE of the confidence against "labelled and inside the
canvas". Optional --neighbor-margin: hinge on each heatmap's logit at its own GT cell vs at the GT cells of the same
side's neighbouring frets (fret +-1), against picking the adjacent wire (wrong fret number).
Validation every --val-every epochs on the val split (inference letterbox): mean / median px error at 640 of labelled
keypoints, confidence accuracy at 0.5, coverage, and slip rate (a keypoint nearer to a neighbouring fret's GT than to
its own). Writes runs/fretnet/<name>/{best,last}.pt + log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro.fretnet.data import PoseDataset  # noqa: E402
from gro.fretnet.model import STRIDE, FretNet, save  # noqa: E402


def neighbor_margin(hm: torch.Tensor, k: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Mean hinge relu(margin - (hm_k[gt_k] - hm_k[gt_j])) over labelled pairs j = k +- 2 (same side, fret +- 1)."""
    b, n, h, w = hm.shape
    vis = k[..., 2] > 0
    ix = ((k[..., 0] + 0.5) / STRIDE - 0.5).round().long().clamp(0, w - 1)
    iy = ((k[..., 1] + 0.5) / STRIDE - 0.5).round().long().clamp(0, h - 1)
    cell = iy * w + ix  # (B, K)
    flat = hm.float().flatten(2)  # (B, K, HW)
    own = flat.gather(2, cell[..., None])[..., 0]
    terms, pairs = [], 0
    for d in (2, -2):
        src = slice(max(0, -d), n - max(0, d))  # keypoint k
        dst = slice(max(0, d), n - max(0, -d))  # its neighbour k + d
        at_nb = flat[:, src].gather(2, cell[:, dst, None])[..., 0]
        ok = vis[:, src] & vis[:, dst]
        terms.append((torch.relu(margin - (own[:, src] - at_nb)) * ok).sum())
        pairs += ok.sum()
    return sum(terms) / pairs.clamp_min(1)


def losses(net: FretNet, hm: torch.Tensor, conf: torch.Tensor, k: torch.Tensor, sigma: float) -> dict:
    b, n, h, w = hm.shape
    vis = k[..., 2] > 0  # (B, K)
    cx = (k[..., 0] + 0.5) / STRIDE - 0.5
    cy = (k[..., 1] + 0.5) / STRIDE - 0.5
    gx = torch.arange(w, device=hm.device, dtype=torch.float32)
    gy = torch.arange(h, device=hm.device, dtype=torch.float32)
    tx = torch.exp(-((gx.view(1, 1, 1, w) - cx[..., None, None]) ** 2) / (2 * sigma**2))
    ty = torch.exp(-((gy.view(1, 1, h, 1) - cy[..., None, None]) ** 2) / (2 * sigma**2))
    target = (tx * ty).flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = torch.log_softmax(hm.float().flatten(2), -1)
    ce = -(target * logp).sum(-1)
    xy = net.locate(hm.float())
    l1 = ((xy - k[..., :2]).abs().sum(-1)) / STRIDE
    nv = vis.sum().clamp_min(1)
    return {
        "ce": (ce * vis).sum() / nv,
        "l1": (l1 * vis).sum() / nv,
        "bce": F.binary_cross_entropy_with_logits(conf.float(), vis.float()),
        "nb": neighbor_margin(hm, k),
    }


@torch.no_grad()
def validate(net: FretNet, loader: DataLoader, device: str) -> dict:
    net.eval()
    errs, correct, total, covered, nvis, slips, slip_n = [], 0, 0, 0, 0, 0, 0
    for x, k in loader:
        x, k = x.to(device, non_blocking=True).float() / 255, k.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm, conf = net(x.contiguous(memory_format=torch.channels_last))
        xy = net.locate(hm.float())
        vis = k[..., 2] > 0
        p = torch.sigmoid(conf.float()) > 0.5
        errs.append(torch.linalg.norm(xy - k[..., :2], dim=-1)[vis].cpu())
        correct += (p == vis).sum().item()
        total += vis.numel()
        covered += (p & vis).sum().item()
        nvis += vis.sum().item()
        own = torch.linalg.norm(xy - k[..., :2], dim=-1)
        for d in (2, -2):  # nearer to a neighbouring fret's GT (same side) than to its own: a numbering slip
            src, dst = slice(max(0, -d), 50 - max(0, d)), slice(max(0, d), 50 - max(0, -d))
            ok = vis[:, src] & vis[:, dst]
            nb = torch.linalg.norm(xy[:, src] - k[:, dst, :2], dim=-1)
            slips += ((nb < own[:, src]) & ok).sum().item()
            slip_n += ok.sum().item()
    e = torch.cat(errs).numpy()
    net.train()
    return {
        "err_mean_px": float(e.mean()),
        "err_median_px": float(np.median(e)),
        "err_p95_px": float(np.percentile(e, 95)),
        "conf_acc": correct / total,
        "coverage": covered / max(nvis, 1),
        "slip_rate": slips / max(slip_n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian target sigma in heatmap cells (4 px)")
    ap.add_argument("--l1", type=float, default=0.5)
    ap.add_argument("--neighbor-margin", type=float, default=0.0, help="weight of the neighbour-fret hinge loss")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--init", type=Path, default=None, help="start from a FretNet checkpoint (fine-tune)")
    a = ap.parse_args()
    out = ROOT / "runs/fretnet" / a.name
    if out.exists():
        ap.error(f"{out} exists; choose a new --name")
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    train = PoseDataset(a.data, "train", train=True, aug={"scale": a.scale, "degrees": a.degrees})
    val = PoseDataset(a.data, "val", train=False)
    tl = DataLoader(train, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)  # fmt: skip
    vl = DataLoader(val, 64, num_workers=a.workers // 2, pin_memory=True, persistent_workers=True)

    net = FretNet(a.arch, pretrained=a.init is None)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True)["state_dict"])
    net = net.to(device).to(memory_format=torch.channels_last)
    backbone = [p for n, p in net.named_parameters() if n.startswith(("stem", "layers"))]
    head = [p for n, p in net.named_parameters() if not n.startswith(("stem", "layers"))]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": a.lr * a.backbone_lr_mult}, {"params": head, "lr": a.lr}], weight_decay=a.wd
    )
    base = [g["lr"] for g in opt.param_groups]
    steps = a.epochs * len(tl)
    warm = min(500, steps // 10)
    best, step = float("inf"), 0
    log = (out / "log.jsonl").open("a")
    for ep in range(a.epochs):
        t0, agg = time.time(), {"ce": 0.0, "l1": 0.0, "bce": 0.0, "nb": 0.0}
        for x, k in tl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
            for g, lr in zip(opt.param_groups, base, strict=True):
                g["lr"] = lr * max(f, 0.01)
            x = (x.to(device, non_blocking=True).float() / 255).contiguous(memory_format=torch.channels_last)
            k = k.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm, conf = net(x)
            ls = losses(net, hm, conf, k, a.sigma)
            loss = ls["ce"] + a.l1 * ls["l1"] + ls["bce"] + a.neighbor_margin * ls["nb"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            for key in agg:
                agg[key] += ls[key].item()
            step += 1
        rec = {"epoch": ep + 1, "sec": round(time.time() - t0, 1), **{k: v / len(tl) for k, v in agg.items()}}
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.epochs:
            rec |= validate(net, vl, device)
            if rec["err_mean_px"] < best:
                best = rec["err_mean_px"]
                save(net, out / "best.pt", epoch=ep + 1, **{k: v for k, v in rec.items() if k.startswith("err")})
        save(net, out / "last.pt", epoch=ep + 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
