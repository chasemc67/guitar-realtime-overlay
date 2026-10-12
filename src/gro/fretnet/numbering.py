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
    def __init__(self, width=1024, pretrained=False, mode="offset"):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.mode = mode
        if mode == "offset":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 16)), nn.Flatten(), nn.Dropout(0.25), nn.Linear(512 * 16, 2 * SHIFT + 1)
            )
        elif mode == "sequence":
            self.projection = nn.Conv1d(512, 256, 3, padding=1)
            self.position = nn.Parameter(torch.zeros(1, 25, 256))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.1, batch_first=True)
            self.context = nn.TransformerEncoder(layer, 2)
            self.head = nn.Linear(256, 25)
        else:
            raise ValueError(f"unknown numbering head: {mode}")
        self.width = width
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        feature = self.features((x.float() / 255 - self.mean) / self.std)
        if self.mode == "offset":
            return self.head(feature)
        # Sample tokens at fitted wires 0..24 in the whole-neck strip, retaining context.
        feature = feature.mean(2, keepdim=True)
        gx = 2 * (torch.arange(25, device=x.device, dtype=feature.dtype) - LO) / (HI - LO) - 1
        grid = torch.stack((gx, torch.zeros_like(gx)), -1)[None, None].expand(len(x), 1, 25, 2)
        tokens = torch.nn.functional.grid_sample(feature, grid, align_corners=True)[:, :, 0]
        tokens = self.projection(tokens).transpose(1, 2) + self.position
        return self.head(self.context(tokens))


def load_numbering(path, device="cpu"):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c.get("kind") != "gro_numbering":
        raise ValueError("not a numbering checkpoint")
    net = NumberingNet(c["width"], mode=c.get("mode", "offset"))
    net.load_state_dict(c["state_dict"])
    return net.to(device).eval()


def renumber(k, offset):
    """Move slot labels only: preserve coordinates/confidences; drop out-of-range slots."""
    out = np.zeros_like(k)
    for i in range(25):
        if 0 <= i + offset < 25:
            out[2 * (i + offset) : 2 * (i + offset) + 2] = k[2 * i : 2 * i + 2]
    return out


def reassign_wires(k, model, offset=0, tolerance=0.45, reject_ambiguous=False):
    """Label each observed point by its fit coordinate, preserving its actual pixel location.

    Ambiguous points retain the stage-1 slot, or are dropped with reject_ambiguous.
    Collisions retain the more confident
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
        reliable = np.isfinite(f[i]) and abs(f[i] - nearest) < tolerance
        if reject_ambiguous and not reliable:
            continue
        assigned = nearest if reliable else i // 2
        target = assigned + offset
        if 0 <= target < 25:
            j = 2 * target + i % 2
            if point[2] > out[j, 2]:
                out[j] = point
    return out


def offset_summary(p):
    """Summarise a per-wire absolute sequence as its most supported global correction."""
    if p.ndim == 1:
        j = int(p.argmax())
        return j - SHIFT, float(p[j])
    absolute = p.argmax(-1)
    differences = absolute - np.arange(len(p))
    values, counts = np.unique(differences, return_counts=True)
    offset = int(values[counts.argmax()])
    confidence = float(p.max(-1)[differences == offset].mean())
    return offset, confidence


def sequence_renumber(k, model, probabilities, threshold):
    """Per-point confidence fallback; sequence columns are indexed by fitted wire position."""
    from gro.geometry import apply_h

    plane = apply_h(np.linalg.inv(model.H), k[:, :2])
    with np.errstate(invalid="ignore", divide="ignore"):
        fret = -12 * np.log2(1 - plane[:, 0])
    out = np.zeros_like(k)
    for i, point in enumerate(k):
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        target = i // 2
        if np.isfinite(fret[i]):
            column = int(round(fret[i]))
            if 0 <= column < 25 and abs(fret[i] - column) < 0.45:
                p = probabilities[column]
                if p.max() >= threshold:
                    target = int(p.argmax())
        j = 2 * target + i % 2
        if point[2] > out[j, 2]:
            out[j] = point
    return out


class NumberingRefiner:
    def __init__(self, path, device="cpu", threshold=0.8, reassign=False, reject_ambiguous=False):
        if not 0 <= threshold <= 1:
            raise ValueError("numbering confidence must be in [0, 1]")
        self.net = load_numbering(path, device)
        self.device, self.threshold = device, threshold
        self.reassign = reassign
        self.reject_ambiguous = reject_ambiguous
        self.last = {}

    def probabilities(self, strip):
        x = torch.from_numpy(np.ascontiguousarray(strip.transpose(2, 0, 1)[None])).to(self.device)
        with torch.inference_mode():
            return self.net(x).softmax(-1)[0].cpu().numpy()

    def classify(self, strip):
        p = self.probabilities(strip)
        return offset_summary(p)

    def __call__(self, rgb, k):
        from gro.geometry import FitError

        self.last = {"offset": 0, "confidence": 0.0, "applied": False, "fit_ok": False}
        try:
            model = fit_fretboard(anchors_from_k(k), ransac_iters=100)
            strip = rectify(rgb, model, self.net.width)
        except (FitError, ValueError, np.linalg.LinAlgError):
            return k
        if getattr(self.net, "mode", "offset") == "sequence":
            p = self.probabilities(strip)
            offset, confidence = offset_summary(p)
            applied = bool((p.max(-1) >= self.threshold).any())
            self.last = {
                "offset": offset,
                "confidence": confidence,
                "applied": applied,
                "fit_ok": True,
                "sequence": p.argmax(-1).tolist(),
            }
            return sequence_renumber(k, model, p, self.threshold)
        offset, confidence = self.classify(strip)
        applied = confidence >= self.threshold
        self.last = {"offset": offset, "confidence": confidence, "applied": applied, "fit_ok": True}
        if applied and self.reassign:
            return reassign_wires(k, model, offset, reject_ambiguous=self.reject_ambiguous)
        return renumber(k, offset) if applied else k
