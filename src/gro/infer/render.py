"""Thin render helpers for TrackState (the state is already in original frame coordinates, so no mirroring here)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from gro.infer.tracker import TrackState
from gro.render import REFERENCE, Style, render_overlay
from gro.theory import ScaleNote


def render_state(
    frame_bgr: np.ndarray,
    state: TrackState,
    notes: Sequence[ScaleNote] = (),
    style: Style = REFERENCE,
    show_kpts: bool = False,
    kpt_thr: float = 0.5,
) -> np.ndarray:
    """Overlay (grid up to the voted last fret + scale notes) on a copy of the frame; the frame itself if not valid."""
    out = render_overlay(frame_bgr, state.model, notes=notes, style=style) if state.valid else frame_bgr.copy()
    if show_kpts:
        for i, (x, y, c) in enumerate(state.kpts):
            if np.isfinite(c) and c > kpt_thr:
                color = (0, 0, 255) if i % 2 == 0 else (0, 200, 255)
                cv2.circle(out, (int(round(x)), int(round(y))), 3, color, 1, cv2.LINE_AA)
    return out


def draw_hud(img: np.ndarray, text: str) -> np.ndarray:
    """Small status line (fps / latency) in the top-left corner, drawn in place."""
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img
