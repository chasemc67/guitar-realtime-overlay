"""Visible landmark-wire localization and confidence-weighted absolute numbering."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from gro.fretnet.numbering import HI, LO, anchors_from_k, reassign_wires, rectify, renumber
from gro.geometry import FitError, apply_h, fit_fretboard

LANDMARKS = (12, 5, 7, 9, 3, 15, 0)
BINS = 113  # quarter-fret spacing over -2..26, plus a separate absent class


class LandmarkNet(nn.Module):
    def __init__(self, width=1024, pretrained=False):
        super().__init__()
        self.width = width
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.local = nn.Sequential(nn.Conv1d(576, 128, 3, padding=1), nn.SiLU(), nn.Conv1d(128, 7, 1))
        self.absent = nn.Linear(576, 7)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        f = self.features((x.float() / 255 - self.mean) / self.std).mean(2)
        heat = nn.functional.interpolate(self.local(f), size=BINS, mode="linear", align_corners=True)
        return torch.cat((heat, self.absent(f.mean(-1))[..., None]), -1)


def load_landmark(path, device="cpu"):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c.get("kind") != "gro_landmark" or tuple(c["landmarks"]) != LANDMARKS:
        raise ValueError("not a supported landmark checkpoint")
    net = LandmarkNet(c["width"])
    net.load_state_dict(c["state_dict"])
    net.metadata = {k: c.get(k) for k in ("epoch", "stats", "seed", "training_data", "selection", "landmarks")}
    return net.to(device).eval()


def decode_landmarks(probabilities):
    """Confidence is posterior mass within +/-0.5 fret of the peak, including absence competition."""
    out = []
    for fret, p in zip(LANDMARKS, probabilities, strict=True):
        peak = int(p[:-1].argmax())
        ids = np.arange(max(0, peak - 2), min(BINS, peak + 3))
        confidence = float(p[ids].sum())
        column = float((ids * p[ids]).sum() / max(confidence, 1e-12))
        out.append(
            dict(
                fret=fret,
                position=LO + column * (HI - LO) / (BINS - 1),
                confidence=confidence,
                presence=float(1 - p[-1]),
            )
        )
    return out


def wire_positions(k, model, reassign=False):
    plane = apply_h(np.linalg.inv(model.H), k[:, :2])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = -12 * np.log2(1 - plane[:, 0])
    wires = []
    for slot in range(25):
        v = f[slot * 2 : slot * 2 + 2]
        valid = (k[slot * 2 : slot * 2 + 2, 2] > 0.5) & np.isfinite(v)
        if valid.any():
            position = float(np.median(v[valid]))
            wires.append((int(round(position)) if reassign else slot, position))
    return wires


def vote_landmarks(detections, wires, threshold=0.8, offset_vote=None):
    """Snap to observed wires; strict majority by confidence, ties fall back to zero."""
    votes, used = {}, []
    for d in detections:
        if d["confidence"] < threshold or not wires:
            continue
        slot, position = min(wires, key=lambda w: abs(w[1] - d["position"]))
        if abs(position - d["position"]) > 0.75:
            continue
        offset = d["fret"] - slot
        votes[offset] = votes.get(offset, 0.0) + d["confidence"]
        used.append(dict(fret=d["fret"], slot=slot, offset=offset, confidence=d["confidence"]))
    if offset_vote is not None:
        offset, confidence = offset_vote
        if confidence >= threshold:
            votes[offset] = votes.get(offset, 0.0) + confidence
    if not votes:
        return 0, 0.0, False, used
    winner = max(votes, key=votes.get)
    support = votes[winner] / sum(votes.values())
    applied = support > 0.5
    return int(winner) if applied else 0, float(support), applied, used


class LandmarkRefiner:
    def __init__(self, path, device="cpu", threshold=0.8, reassign=False, reject_ambiguous=False, numbering=None):
        if not 0 <= threshold <= 1:
            raise ValueError("landmark confidence must be in [0, 1]")
        self.net = load_landmark(path, device)
        self.device, self.threshold = device, threshold
        self.reassign, self.reject_ambiguous = reassign, reject_ambiguous
        self.offset_model = None
        if numbering is not None:
            from gro.fretnet.numbering import NumberingRefiner

            self.offset_model = NumberingRefiner(numbering, device, threshold)
            if self.offset_model.net.mode != "offset":
                raise ValueError("landmark fusion requires an offset classifier")
        self.last = {}

    def detect(self, strip):
        x = torch.from_numpy(np.ascontiguousarray(strip.transpose(2, 0, 1)[None])).to(self.device)
        with torch.inference_mode():
            return decode_landmarks(self.net(x).softmax(-1)[0].cpu().numpy())

    def __call__(self, rgb, k, fit_model=None):
        self.last = dict(offset=0, confidence=0.0, applied=False, fit_ok=False, landmarks=[])
        try:
            model = fit_model if fit_model is not None else fit_fretboard(anchors_from_k(k), ransac_iters=100)
            detections = self.detect(rectify(rgb, model, self.net.width))
        except (FitError, ValueError, np.linalg.LinAlgError):
            return k
        self.fitted_model = model
        offset_vote = None
        if self.offset_model is not None:
            offset_vote = self.offset_model.classify(rectify(rgb, model, self.offset_model.net.width))
        offset, confidence, applied, votes = vote_landmarks(
            detections, wire_positions(k, model, self.reassign), self.threshold, offset_vote
        )
        self.last = dict(
            offset=offset,
            confidence=confidence,
            applied=applied,
            fit_ok=True,
            landmarks=detections,
            votes=votes,
            offset_vote=offset_vote,
        )
        if self.reassign:
            return reassign_wires(k, model, offset, reject_ambiguous=self.reject_ambiguous)
        return renumber(k, offset) if applied else k
