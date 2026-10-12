"""Temporal smoothing of the fitted fretboard (One-Euro filter on the homography, PLAN §3.4).

The per-frame fit is represented by the image positions of fixed canonical control points (wires 0/6/12/18/last on
both outer strings). Each coordinate gets a One-Euro filter (Casiez et al. 2012) whose cutoff rises with speed, so
static segments are smoothed hard while fast moves pass with little lag. Speeds below `speed_floor` (the apparent speed
of per-frame detector noise) don't raise the cutoff. All points share one cutoff, computed from
the median speed of the in-view control points in fret widths per second, so the grid never shears. The smoothed H is
refit from the filtered points by DLT.
"""

from __future__ import annotations

import numpy as np

from gro.geometry import FretboardModel, dlt, fret_u, plane_point

CONTROL_FRETS = (0, 6, 12, 18)


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroModel:
    def __init__(
        self, fps: float, min_cutoff: float = 0.4, beta: float = 3.0, d_cutoff: float = 1.0, speed_floor: float = 2.0
    ):
        self.dt = 1.0 / fps
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.speed_floor = speed_floor  # fret widths/s of apparent speed that is detector noise, not motion
        self.x = None  # filtered control points (K, 2)
        self.dx = 0.0  # filtered speed, fret widths / s
        self.key = None

    def _controls(self, model: FretboardModel):
        frets = np.array([*CONTROL_FRETS, model.n_frets], float)
        f = np.repeat(frets, 2)
        y = np.tile([0.0, 1.0], len(frets))
        canon = plane_point(fret_u(f), y, model.tau)
        img = model.project(fret_u(f), y)
        fw = model.fret_widths_px(np.minimum(f, model.n_frets - 1), y)
        return canon, img, fw

    def reset(self):
        self.x, self.dx, self.key = None, 0.0, None

    def update(self, model: FretboardModel, shape, dt: float | None = None) -> FretboardModel:
        """Filter one fit. `dt` (s since the previous update) defaults to 1 / fps; pass it when frames are dropped."""
        dt = self.dt if dt is None or not dt > 0 else dt
        canon, img, fw = self._controls(model)
        key = (model.n_frets, model.tau)
        if self.x is None or key != self.key or not np.isfinite(img).all():
            self.x, self.dx, self.key = img, 0.0, key
            return model
        h, w = shape[:2]
        inview = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        sel = inview if inview.sum() >= 2 else np.ones(len(img), bool)
        speed = float(np.median(np.linalg.norm(img[sel] - self.x[sel], axis=1) / np.maximum(fw[sel], 1e-3))) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * speed + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * max(0.0, self.dx - self.speed_floor)
        a = _alpha(cutoff, dt)
        self.x = a * img + (1 - a) * self.x
        out = FretboardModel(**{**model.__dict__})
        H = dlt(canon, self.x)
        out.H = H / H[2, 2]
        return out
