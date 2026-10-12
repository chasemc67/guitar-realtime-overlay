"""Whole-neck numbering from rectified RGB context; no detector weights are used here."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from gro.anchors import Anchor
from gro.geometry import fit_fretboard

SHIFT = 3
LO, HI, HEIGHT = -2.0, 26.0, 128


def anchors_from_k(k):
    return [
        Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
        for i, (x, y, c) in enumerate(k)
        if c > 0.5 and np.isfinite([x, y, c]).all()
    ]


def rectify(rgb, model, width=1024, lo=LO, hi=HI, height=HEIGHT):
    """Equal-fret columns, nut side left, with two frets of context beyond each end."""
    f, y = np.meshgrid(np.linspace(lo, hi, width), np.linspace(-0.35, 1.35, height))
    xy = model.project_fy(f, y).astype(np.float32)
    if not np.isfinite(xy).all():
        raise ValueError("nonfinite strip projection")
    return cv2.remap(rgb, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR, borderValue=(114,) * 3)


class NumberingNet(nn.Module):
    def __init__(self, width=1024, pretrained=False):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 16)), nn.Flatten(), nn.Dropout(0.25), nn.Linear(512 * 16, 7))
        self.width = width
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        return self.head(self.features((x.float() / 255 - self.mean) / self.std))


def load_numbering(path, device="cpu"):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c.get("kind") != "gro_numbering":
        raise ValueError("not a numbering checkpoint")
    net = NumberingNet(c["width"])
    net.load_state_dict(c["state_dict"])
    return net.to(device).eval()


def renumber(k, offset):
    """Move slot labels only: preserve coordinates/confidences; drop out-of-range slots."""
    out = np.zeros_like(k)
    for i in range(25):
        if 0 <= i + offset < 25:
            out[2 * (i + offset) : 2 * (i + offset) + 2] = k[2 * i : 2 * i + 2]
    return out


def reassign_wires(k, model, offset=0, tolerance=0.45):
    """Label each observed point by its fit coordinate, preserving its actual pixel location.

    Ambiguous points retain the stage-1 slot. Collisions retain the more confident
    observation; assignments outside the representable 0..24 range are omitted.
    """
    from gro.geometry import apply_h

    plane = apply_h(np.linalg.inv(model.H), k[:, :2])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = -12 * np.log2(1 - plane[:, 0])
    out = np.zeros_like(k)
    for i, point in enumerate(k):
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        nearest = int(round(f[i])) if np.isfinite(f[i]) else i // 2
        assigned = nearest if np.isfinite(f[i]) and abs(f[i] - nearest) < tolerance else i // 2
        target = assigned + offset
        if 0 <= target < 25:
            j = 2 * target + i % 2
            if point[2] > out[j, 2]:
                out[j] = point
    return out


class NumberingRefiner:
    def __init__(self, path, device="cpu", threshold=0.8, reassign=False):
        if not 0 <= threshold <= 1:
            raise ValueError("numbering confidence must be in [0, 1]")
        self.net = load_numbering(path, device)
        self.device, self.threshold = device, threshold
        self.reassign = reassign
        self.last = {}

    def classify(self, strip):
        x = torch.from_numpy(np.ascontiguousarray(strip.transpose(2, 0, 1)[None])).to(self.device)
        with torch.inference_mode():
            p = self.net(x).softmax(-1)[0].cpu().numpy()
        j = int(p.argmax())
        return j - SHIFT, float(p[j])

    def __call__(self, rgb, k):
        from gro.geometry import FitError

        self.last = {"offset": 0, "confidence": 0.0, "applied": False, "fit_ok": False}
        try:
            model = fit_fretboard(anchors_from_k(k), ransac_iters=100)
            strip = rectify(rgb, model, self.net.width)
        except (FitError, ValueError, np.linalg.LinAlgError):
            return k
        offset, confidence = self.classify(strip)
        applied = confidence >= self.threshold
        self.last = {"offset": offset, "confidence": confidence, "applied": applied, "fit_ok": True}
        if applied and self.reassign:
            return reassign_wires(k, model, offset)
        return renumber(k, offset) if applied else k
