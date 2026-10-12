"""Letterbox in, decode + unletterbox out: the model-format-independent ends of the pose network.

Raw head output (every backend): float32 (155, 8400) = per anchor [cx, cy, w, h, score, x0, y0, c0, ..., x49, y49, c49]
in letterboxed input pixels, score and keypoint confidences already sigmoid-ed (ultralytics export layout, nms=False).
Keypoint k = 2 * fret + side (side 0 = bass / low-E edge, 1 = treble / high-E edge), frets 0 (nut) .. 24.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

N_KPTS = 50
PAD_VALUE = 114


@dataclass(frozen=True)
class Letterbox:
    """Placement of a (h, w) frame inside the square network input (same rounding as ultralytics LetterBox)."""

    size: int
    h: int
    w: int
    new_w: int
    new_h: int
    left: int
    top: int

    @classmethod
    def for_shape(cls, shape, size: int = 640) -> Letterbox:
        h, w = shape[:2]
        r = min(size / h, size / w)
        nw, nh = round(w * r), round(h * r)
        return cls(size, h, w, nw, nh, round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> (size, size, 3) BGR uint8, resized (bilinear) and padded with 114."""
        img = frame
        if (self.new_w, self.new_h) != (self.w, self.h):
            img = cv2.resize(frame, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        bottom = self.size - self.new_h - self.top
        right = self.size - self.new_w - self.left
        return cv2.copyMakeBorder(img, self.top, bottom, self.left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)

    def to_frame(self, xy: np.ndarray, clip: bool = True) -> np.ndarray:
        """Letterboxed input px -> frame px (..., 2). `clip` clamps to the frame like ultralytics' scale_coords."""
        out = np.array(xy, dtype=np.float32, copy=True)
        out[..., 0] = (out[..., 0] - self.left) / (self.new_w / self.w)
        out[..., 1] = (out[..., 1] - self.top) / (self.new_h / self.h)
        if clip:
            out[..., 0] = out[..., 0].clip(0, self.w)
            out[..., 1] = out[..., 1].clip(0, self.h)
        return out


def decode(raw: np.ndarray, conf_thr: float = 0.25) -> tuple[np.ndarray, float, np.ndarray]:
    """Best detection from a raw (155, A) or (1, 155, A) head output.

    Returns (kpts (50, 3) x, y, conf in letterboxed px, board score, box (4,) cx, cy, w, h). With one class the
    best-scoring anchor is the box NMS would keep first, so no NMS is needed. Below `conf_thr`: NaN kpts, score kept.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    i = int(np.argmax(raw[4]))
    score = float(raw[4, i])
    if score < conf_thr:
        return np.full((N_KPTS, 3), np.nan, np.float32), score, np.full(4, np.nan, np.float32)
    return raw[5:, i].reshape(N_KPTS, 3).copy(), score, raw[:4, i].copy()
