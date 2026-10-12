"""Train the RTMPose-style top-down model (gro.fretnet.rtmpose; CSPNeXt + RTMCC/SimCC head, no MMPose code).

    .venv/bin/python scripts/train_rtmpose.py --name v6p_cspm --pretrained ~/gro/weights/cspnext-m_in1k.pth

Each sample is an affine crop of the source frame (1280x720 for most of scale-v1aug) around the label's keypoint box.
Strong augmentation, because the full-frame SimCC run (v5s) learnt the training necks' positions and did not generalise:
box jitter (centre +-15 % of the box, expand 1.1-1.6), rotation +-15 deg, partial boxes (with p 0.25 the box covers only
a random run of frets; keypoints outside the crop become unlabelled), HSV, blur, random erasing. Loss: cross-entropy of
each axis' 1D bins against a Gaussian of --sigma bins, L1 of the located px / 4, confidence BCE (labelled and inside).
Val: GT box, expand 1.25, no jitter; errors in crop px. Writes runs/rtmpose/<name>/{best,last}.pt + log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro.fretnet.data import read_label  # noqa: E402
from gro.fretnet.rtmpose import RTMPoseNet, box_affine, kpt_box, save  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, root: Path, split: str, w: int, h: int, train: bool):
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        self.labels = root / "labels" / split
        self.w, self.h, self.train = w, h, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        img = cv2.imread(str(path))
        ih, iw = img.shape[:2]
        k = read_label(self.labels / f"{path.stem}.txt", iw, ih)
        vis = k[:, 2] > 0
        rng = np.random.default_rng()
        box = kpt_box(k[:, :2], vis)
        if box is None:
            box = np.array([0, 0, iw, ih], np.float32)
        expand, angle = 1.25, 0.0
        if self.train:
            if rng.random() < 0.25 and vis.sum() >= 8:  # partial box: a run of frets only
                frets = np.unique(np.nonzero(vis)[0] // 2)
                n = rng.integers(max(3, len(frets) // 3), len(frets) + 1)
                s = rng.integers(0, len(frets) - n + 1)
                sel = vis & np.isin(np.arange(50) // 2, frets[s : s + n])
                box = kpt_box(k[:, :2], sel)
            bw, bh = box[2] - box[0], box[3] - box[1]
            d = rng.uniform(-0.15, 0.15, 2) * [bw, bh]
            box = box + np.array([d[0], d[1], d[0], d[1]], np.float32)
            expand, angle = rng.uniform(1.1, 1.6), rng.uniform(-15, 15)
        m = box_affine(box, self.w, self.h, expand, angle)
        out = cv2.warpAffine(img, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        k[:, :2] = k[:, :2] @ m[:, :2].T + m[:, 2]
        inside = (k[:, 0] >= 0) & (k[:, 0] < self.w) & (k[:, 1] >= 0) & (k[:, 1] < self.h)
        k[:, 2] = (vis & inside).astype(np.float32)
        if self.train:
            out = photometric(out, rng)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))), torch.from_numpy(k)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gains = rng.uniform(-1, 1, 3) * [0.015, 0.6, 0.4] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut = [((x * gains[0]) % 180), np.clip(x * gains[1], 0, 255), np.clip(x * gains[2], 0, 255)]
    hsv = cv2.merge([cv2.LUT(c, t.astype(np.uint8)) for c, t in zip((hue, sat, val), lut, strict=True)])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.5:
        h, w = img.shape[:2]
        for _ in range(rng.integers(1, 5)):
            ew, eh = int(rng.uniform(0.03, 0.15) * w), int(rng.uniform(0.05, 0.3) * h)
            x0, y0 = rng.integers(0, w - ew), rng.integers(0, h - eh)
            img[y0 : y0 + eh, x0 : x0 + ew] = rng.integers(0, 255, 3)
    return img


def losses(net: RTMPoseNet, sx, sy, conf, k, sigma: float) -> dict:
    vis = k[..., 2] > 0
    nv = vis.sum().clamp_min(1)
    ce = 0.0
    for logits, c in ((sx, k[..., 0]), (sy, k[..., 1])):
        g = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        t = torch.exp(-((g - ((c[..., None] + 0.5) * net.split - 0.5)) ** 2) / (2 * sigma**2))
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-12)
        ce = ce + (-(t * torch.log_softmax(logits.float(), -1)).sum(-1) * vis).sum() / nv
    xy = torch.stack([net.locate1d(sx.float()), net.locate1d(sy.float())], -1)
    l1 = (xy - k[..., :2]).abs().sum(-1) / 4
    return {"ce": ce, "l1": (l1 * vis).sum() / nv, "bce": F.binary_cross_entropy_with_logits(conf.float(), vis.float())}


@torch.no_grad()
def validate(net, loader, device) -> dict:
    net.eval()
    errs, covered, nvis, slips, slip_n = [], 0, 0, 0, 0
    for x, k in loader:
        x, k = x.to(device).float() / 255, k.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xy, conf = net.predict(x)
        xy = xy.float()
        vis = k[..., 2] > 0
        own = torch.linalg.norm(xy - k[..., :2], dim=-1)
        errs.append(own[vis].cpu())
        covered += ((torch.sigmoid(conf) > 0.5) & vis).sum().item()
        nvis += vis.sum().item()
        for d in (2, -2):
            src, dst = slice(max(0, -d), 50 - max(0, d)), slice(max(0, d), 50 - max(0, -d))
            ok = vis[:, src] & vis[:, dst]
            slips += ((torch.linalg.norm(xy[:, src] - k[:, dst, :2], dim=-1) < own[:, src]) & ok).sum().item()
            slip_n += ok.sum().item()
    e = torch.cat(errs).numpy()
    net.train()
    return {"err_mean_px": float(e.mean()), "err_median_px": float(np.median(e)), "err_p95_px": float(np.percentile(e, 95)),
            "coverage": covered / max(nvis, 1), "slip_rate": slips / max(slip_n, 1)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--pretrained", type=Path, required=True, help="CSPNeXt ImageNet checkpoint (MMDet layout)")
    ap.add_argument("--size", default="m")
    ap.add_argument("--w", type=int, default=768)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=4.0, help="SimCC Gaussian sigma in bins (split 2)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--oversample-high", type=float, default=2.0)
    a = ap.parse_args()
    out = ROOT / "runs/rtmpose" / a.name
    if out.exists():
        ap.error(f"{out} exists")
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    train = CropDataset(a.data, "train", a.w, a.h, True)
    val = CropDataset(a.data, "val", a.w, a.h, False)
    last = [max((f for f, v in enumerate(read_label(train.labels / f"{p.stem}.txt", 1, 1)[1::2, 2]) if v > 0), default=-1)
            for p in train.images]  # fmt: skip
    wts = torch.tensor([a.oversample_high if f >= 23 else 1.0 for f in last], dtype=torch.double)
    tl = DataLoader(train, a.batch, sampler=WeightedRandomSampler(wts, len(train), replacement=True),
                    num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)  # fmt: skip
    vl = DataLoader(val, 64, num_workers=a.workers // 2, persistent_workers=True)
    net = RTMPoseNet(a.size, str(a.pretrained), a.w, a.h).to(device)
    bb = [p for n, p in net.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * 0.5}, {"params": hd, "lr": a.lr}], weight_decay=a.wd)
    base = [g["lr"] for g in opt.param_groups]
    steps = a.epochs * len(tl)
    warm = min(1000, steps // 10)
    best, step = float("inf"), 0
    log = (out / "log.jsonl").open("a")
    for ep in range(a.epochs):
        t0, agg = time.time(), {"ce": 0.0, "l1": 0.0, "bce": 0.0}
        for x, k in tl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
            for g, lr in zip(opt.param_groups, base, strict=True):
                g["lr"] = lr * max(f, 0.01)
            x = x.to(device, non_blocking=True).float() / 255
            k = k.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x)
            ls = losses(net, *pred, k, a.sigma)
            loss = ls["ce"] + 0.5 * ls["l1"] + ls["bce"]
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
