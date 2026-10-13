"""Train visible landmark locations on guarded scale-v1aug GT strips only."""

from __future__ import annotations

import argparse
import hashlib
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
from train_numbering import cache_one, guarded_images  # noqa: E402

from gro.fretnet.data import read_label  # noqa: E402
from gro.fretnet.landmark import BINS, LANDMARKS, LandmarkNet  # noqa: E402
from gro.fretnet.numbering import HI, LO, SHIFT  # noqa: E402


class LandmarkStrips(Dataset):
    def __init__(self, paths, labels, width, train):
        self.paths, self.width, self.train = paths, width, train
        self.present = []
        for p in paths:
            k = read_label(labels / (p.stem + ".txt"), 1, 1)
            visible = (k[:, 2] > 0) & (k[:, 0] >= 0) & (k[:, 0] < 1) & (k[:, 1] >= 0) & (k[:, 1] < 1)
            self.present.append(visible.reshape(25, 2).any(1)[list(LANDMARKS)])
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng(17 + i + self.epoch * len(self.paths))
        shift = float(rng.uniform(-SHIFT, SHIFT))
        img = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        x0 = (SHIFT + 1 + shift) * 48
        quad = np.array([[x0, 0], [x0 + (HI - LO) * 48, 0], [x0 + (HI - LO) * 48, 159], [x0, 159]], np.float32)
        if self.train:
            quad += rng.normal(0, [8, 6], quad.shape).astype(np.float32)
        dst = np.array([[0, 0], [self.width - 1, 0], [self.width - 1, 127], [0, 127]], np.float32)
        warp = cv2.getPerspectiveTransform(quad, dst)
        out = cv2.warpPerspective(img, warp, (self.width, 128), borderValue=(114,) * 3)
        # Apply the identical jittered warp to each GT wire's center; no target drift.
        xy = np.array([[(f - (LO - SHIFT - 1)) * 48, 79.5] for f in LANDMARKS], np.float32)
        projected = cv2.perspectiveTransform(xy[None], warp)[0]
        positions = projected[:, 0] / (self.width - 1) * (BINS - 1)
        present = self.present[i] & (positions >= 0) & (positions <= BINS - 1)
        target = np.zeros((len(LANDMARKS), BINS + 1), np.float32)
        heat = np.exp(-0.5 * (np.arange(BINS)[None] - positions[:, None]) ** 2)
        heat /= np.maximum(heat.sum(1, keepdims=True), 1e-12)
        target[present, :BINS] = heat[present]
        target[~present, BINS] = 1
        if self.train:
            out = np.clip(out.astype(float) * rng.uniform(0.7, 1.3, (1, 1, 3)) + rng.uniform(-15, 15), 0, 255).astype(
                np.uint8
            )
            if rng.random() < 0.2:
                out = cv2.GaussianBlur(out, (3, 3), 0.8)
            if rng.random() < 0.3:
                x = int(rng.integers(0, self.width - 80))
                out[:, x : x + 80] = rng.integers(30, 200, 3)
        return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))), torch.from_numpy(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/datasets/numbering-strips")
    ap.add_argument("--name", default="v8_landmark_1024")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(1)
    torch.manual_seed(17)
    cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
    loaders, hashes = {}, {}
    for split in ("train", "val"):
        images = guarded_images(a.data, split, cfg)
        labels = a.data / "labels" / split
        if any(not (labels / (p.stem + ".txt")).is_file() for p in images):
            raise ValueError("missing labels")
        hashes[split] = hashlib.sha256(
            b"".join(p.name.encode() + (labels / (p.stem + ".txt")).read_bytes() for p in images)
        ).hexdigest()
        dest = a.cache / split
        dest.mkdir(parents=True, exist_ok=True)
        tasks = [(str(p), str(labels / (p.stem + ".txt")), str(dest / p.name)) for p in images]
        with ProcessPoolExecutor(a.workers) as pool:
            ok = list(pool.map(cache_one, tasks, chunksize=16))
        paths = [Path(t[2]) for t, valid in zip(tasks, ok, strict=True) if valid]
        print(f"{split}: {len(paths)}/{len(images)} valid fits", flush=True)
        loaders[split] = DataLoader(
            LandmarkStrips(paths, labels, a.width, split == "train"),
            batch_size=a.batch,
            shuffle=split == "train",
            num_workers=a.workers,
        )
    out = ROOT / "runs/fretnet" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a) | {"split_hashes": hashes}, default=str, indent=2))
    device = torch.device(a.device)
    net = LandmarkNet(a.width, pretrained=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        start, stats = time.time(), {"epoch": epoch}
        for split, loader in loaders.items():
            train = split == "train"
            net.train(train)
            loader.dataset.epoch = epoch if train else 0
            loss_sum = count = correct = 0
            with torch.set_grad_enabled(train):
                for x, target in loader:
                    x, target = x.to(device), target.to(device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits = net(x)
                        losses = -(target * logits.log_softmax(-1)).sum(-1)
                        weights = losses.new_tensor([2, 1, 1, 1, 1, 1, 1])
                        loss = (losses * weights).mean()
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                        opt.step()
                    pred, truth = logits.argmax(-1), target.argmax(-1)
                    good = ((pred - truth).abs() <= 2) & (pred < BINS) & (truth < BINS)
                    good |= (pred == BINS) & (truth == BINS)
                    correct += int(good.sum())
                    count += truth.numel()
                    loss_sum += float(loss.detach()) * truth.numel()
            stats[split + "_loss"] = loss_sum / count
            stats[split + "_accuracy"] = correct / count
        sched.step()
        stats["seconds"] = time.time() - start
        checkpoint = dict(
            kind="gro_landmark",
            width=a.width,
            landmarks=list(LANDMARKS),
            epoch=epoch,
            state_dict=net.state_dict(),
            stats=stats,
            seed=17,
            training_data=str(a.data),
            selection="val_loss",
            split_hashes=hashes,
        )
        torch.save(checkpoint, out / "last.pt")
        if stats["val_loss"] < best:
            best = stats["val_loss"]
            torch.save(checkpoint, out / "best.pt")
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(stats) + "\n")
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
