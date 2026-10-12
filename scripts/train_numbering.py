"""Train rectified whole-neck offset classifier on scale-v1aug; held-out IDs fail closed.

Cache GT strips on CPU first, then train a torchvision backbone using random integer offsets,
sub-fret/projective jitter, colour/blur/occlusion augmentation. Select checkpoints by val only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gro.fretnet.data import read_label  # noqa: E402
from gro.fretnet.numbering import HI, LO, SHIFT, NumberingNet, anchors_from_k, rectify  # noqa: E402
from gro.geometry import FitError, fit_fretboard  # noqa: E402


def cache_one(args):
    image, label, dest = map(Path, args)
    cv2.setNumThreads(1)
    if dest.exists():
        return True
    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    k = read_label(label, rgb.shape[1], rgb.shape[0])
    try:
        model = fit_fretboard(anchors_from_k(k), ransac_iters=100)
        strip = rectify(rgb, model, 1729, LO - SHIFT - 1, HI + SHIFT + 1, height=160)
    except (FitError, ValueError, np.linalg.LinAlgError):
        return False
    cv2.imwrite(str(dest), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return True


def guarded_images(data, split, cfg):
    paths = sorted((data / "images" / split).glob("*.jpg"))
    if not paths:
        raise ValueError(f"empty split: {split}")
    forbidden = [c["id"] for c in cfg["clips"]] + ["2026-09-06", "09-06", "anchors_0906"]
    bad = [p.name for p in paths if any(s in p.name for s in forbidden)]
    if bad:
        raise ValueError(f"held-out images in {split}: {bad[:10]}")
    return paths


class Strips(Dataset):
    def __init__(self, paths, width, train, labels):
        self.paths, self.width, self.train = paths, width, train
        self.present = [(read_label(labels / (p.stem + ".txt"), 1, 1)[:, 2].reshape(25, 2) > 0).any(1) for p in paths]

    def __len__(self):
        return len(self.paths) if self.train else len(self.paths) * 7

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        shift = int(rng.integers(-SHIFT, SHIFT + 1)) if self.train else i % 7 - SHIFT
        index = i if self.train else i // 7
        p = self.paths[index]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        # Cached coordinate range -6..30; crop -2..26 plus the sampled integer shift.
        x0 = (SHIFT + 1 + shift) * 48
        x1 = x0 + (HI - LO) * 48
        quad = np.array([[x0, 0], [x1, 0], [x1, 159], [x0, 159]], np.float32)
        if self.train:
            quad += rng.normal(0, [5.0, 5.0], quad.shape).astype(np.float32)
        dst = np.array([[0, 0], [self.width - 1, 0], [self.width - 1, 127], [0, 127]], np.float32)
        out = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(quad, dst), (self.width, 128), borderValue=(114,) * 3
        )
        if self.train:
            out = np.clip(out.astype(float) * rng.uniform(0.7, 1.3, (1, 1, 3)) + rng.uniform(-15, 15), 0, 255).astype(
                np.uint8
            )
            if rng.random() < 0.2:
                out = cv2.GaussianBlur(out, (3, 3), 0.8)
            if rng.random() < 0.3:
                x = int(rng.integers(0, self.width - 80))
                out[:, x : x + 80] = rng.integers(30, 200, 3)
        return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))), shift + SHIFT, self.present[index]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/datasets/numbering-strips")
    ap.add_argument("--name", default="v7num_1024")
    ap.add_argument("--arch", choices=["resnet18", "mobilenet_v3_small"], default="mobilenet_v3_small")
    ap.add_argument("--mode", choices=["offset", "sequence"], default="offset")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
    splits = {}
    for split in ["train", "val"]:
        images = guarded_images(a.data, split, cfg)
        dest = a.cache / split
        dest.mkdir(parents=True, exist_ok=True)
        tasks = [(str(p), str(a.data / "labels" / split / (p.stem + ".txt")), str(dest / p.name)) for p in images]
        with ProcessPoolExecutor(a.workers) as pool:
            ok = list(pool.map(cache_one, tasks, chunksize=16))
        splits[split] = [Path(t[2]) for t, good in zip(tasks, ok, strict=True) if good]
        print(f"{split}: {sum(ok)}/{len(ok)} valid GT fits", flush=True)
    if a.cache_only:
        return
    out = ROOT / "runs/fretnet" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), default=str, indent=2))
    loaders = {
        s: DataLoader(
            Strips(p, a.width, s == "train", a.data / "labels" / s),
            batch_size=a.batch,
            shuffle=s == "train",
            num_workers=a.workers,
            pin_memory=True,
        )
        for s, p in splits.items()
    }
    net = NumberingNet(a.width, pretrained=True, mode=a.mode, arch=a.arch).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = -1
    for epoch in range(1, a.epochs + 1):
        start = time.time()
        stats = {"epoch": epoch}
        for split, loader in loaders.items():
            net.train(split == "train")
            correct = count = total = 0
            with torch.set_grad_enabled(split == "train"):
                for x, y, present in loader:
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    present = present.cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(x)
                        if a.mode == "sequence":
                            target = torch.arange(25, device=y.device)[None] + y[:, None] - SHIFT
                            valid = (target >= 0) & (target < 25)
                            valid &= present.gather(1, target.clamp(0, 24))
                            if not valid.any():
                                continue
                            target = target.masked_fill(~valid, -100)
                            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target)
                        else:
                            target = y
                            valid = torch.ones_like(y, dtype=torch.bool)
                            loss = torch.nn.functional.cross_entropy(logits, y)
                    if split == "train":
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                        opt.step()
                    correct += int(((logits.argmax(-1) == target) & valid).sum())
                    count += int(valid.sum())
                    total += float(loss.detach()) * int(valid.sum())
            stats[split + "_accuracy"] = correct / count
            stats[split + "_loss"] = total / count
        sched.step()
        checkpoint = {
            "kind": "gro_numbering",
            "width": a.width,
            "mode": a.mode,
            "arch": a.arch,
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "stats": stats,
            "seed": 17,
            "training_data": str(a.data),
            "selection": "val_accuracy",
        }
        torch.save(checkpoint, out / "last.pt")
        if stats["val_accuracy"] > best:
            best = stats["val_accuracy"]
            torch.save(checkpoint, out / "best.pt")
        stats["seconds"] = time.time() - start
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(stats) + "\n")
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
