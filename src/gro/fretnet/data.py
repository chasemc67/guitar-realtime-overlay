"""Training data for FretNet: the YOLO-pose export (data/datasets/scale-v1*/{images,labels}/<split>), read with our own
loader. Label line: `0 cx cy w h` + 50 x `(x y v)`, normalised to the image; v > 0 = labelled.

Train: random affine (scale / rotation / translation) straight into the 640 letterbox canvas, HSV jitter, blur, random
erasing. No horizontal flip (the model is right-handed only, D1). Val: the exact inference letterbox (gro.infer).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gro.infer.preprocess import PAD_VALUE, Letterbox

N_KPTS = 50


def read_label(path: Path, w: int, h: int) -> np.ndarray:
    """(50, 3) x, y in px, v (0 = absent)."""
    k = np.zeros((N_KPTS, 3), np.float32)
    if path.exists():
        rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
        if rows:
            v = np.array(rows[0][5:], np.float32).reshape(N_KPTS, 3)
            k[:, 0], k[:, 1], k[:, 2] = v[:, 0] * w, v[:, 1] * h, v[:, 2]
    return k


class PoseDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int = 640, train: bool = True, aug: dict | None = None):
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        self.labels = root / "labels" / split
        self.size, self.train = size, train
        self.aug = {"scale": 0.25, "degrees": 10.0, "translate": 0.1, "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.35,
                    "erasing": 0.4, "blur": 0.1} | (aug or {})  # fmt: skip

    def __len__(self) -> int:
        return len(self.images)

    def _affine(self, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        a, s = self.aug, self.size
        r = s / max(h, w)
        if not self.train:
            return np.array([[r, 0, 0], [0, r, 0]], np.float32)
        scale = r * rng.uniform(1 - a["scale"], 1 + a["scale"])
        ang = math.radians(rng.uniform(-a["degrees"], a["degrees"]))
        tx, ty = rng.uniform(-a["translate"], a["translate"], 2) * s
        c, sn = math.cos(ang) * scale, math.sin(ang) * scale
        m = np.array([[c, -sn, 0], [sn, c, 0]], np.float32)
        m[:, 2] = np.array([s / 2 + tx, s / 2 + ty]) - m[:, :2] @ np.array([w / 2, h / 2])
        return m

    def __getitem__(self, i: int):
        path = self.images[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        k = read_label(self.labels / f"{path.stem}.txt", w, h)
        s = self.size
        if self.train:
            rng = np.random.default_rng()
            m = self._affine(h, w, rng)
            out = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=(PAD_VALUE,) * 3)
            k[:, :2] = k[:, :2] @ m[:, :2].T + m[:, 2]
            out = self._photometric(out, rng)
        else:
            lb = Letterbox.for_shape(img.shape, s)
            out = lb.apply(img)
            k[:, 0] = k[:, 0] * lb.new_w / lb.w + lb.left
            k[:, 1] = k[:, 1] * lb.new_h / lb.h + lb.top
        inside = (k[:, 0] >= 0) & (k[:, 0] < s) & (k[:, 1] >= 0) & (k[:, 1] < s)
        k[:, 2] = ((k[:, 2] > 0) & inside).astype(np.float32)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))  # uint8, scaled on the GPU
        return x, torch.from_numpy(k)

    def _photometric(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a = self.aug
        gains = rng.uniform(-1, 1, 3) * [a["hsv_h"], a["hsv_s"], a["hsv_v"]] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(256, dtype=np.float32)
        lut_h = ((x * gains[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if rng.random() < a["blur"]:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < a["erasing"]:
            s = self.size
            for _ in range(rng.integers(1, 4)):
                ew, eh = (rng.uniform(0.03, 0.15, 2) * s).astype(int)
                x0, y0 = rng.integers(0, s - ew), rng.integers(0, s - eh)
                img[y0 : y0 + eh, x0 : x0 + ew] = rng.integers(0, 255, 3)
        return img
