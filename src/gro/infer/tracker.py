"""Real-time fretboard tracker: frame in, smoothed fretboard homography out (demo v2 pipeline as one class).

    tracker = FretboardTracker("v1_fp16.mlpackage", mirrored=True)
    state = tracker.process(frame_bgr, t)   # t = capture time, seconds
    if state.valid:
        out = render_state(frame_bgr, state, notes=scale_positions("A", "minor pentatonic"))

Per frame: mirror into canonical (right-handed) orientation if `mirrored` (D1) -> letterbox -> backend -> best box
-> keypoints -> anchors (conf > kpt_thr) -> board-end probe fit + ridge test -> FretCountVote -> RANSAC fit up to
the voted last fret (until the vote settles: at most the highest confidently seen fret) -> One-Euro smoothing
(gro.smooth) -> hold the last fit for `hold` missed frames. Everything in the returned TrackState is in *original*
frame pixels (the mirror is composed into H and applied to keypoints).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from gro.anchors import Anchor
from gro.board_end import FretCountVote, frame_last_fret, ridge_scores
from gro.geometry import FitError, FretboardModel, fit_fretboard, mirror_model
from gro.infer.backends import Backend, load_backend
from gro.infer.preprocess import N_KPTS, Letterbox, decode
from gro.smooth import OneEuroModel

END_FIT_MAX_FRET = 21  # board-end probe fit ignores slots that may sit on the pickups of a 21/22-fret guitar


@dataclass
class TrackState:
    """Tracker output for one frame. Coordinates are original-frame pixels (x right, y down)."""

    t: float
    valid: bool  # a (possibly held) fit is available
    model: FretboardModel | None = None  # smoothed fit; model.H maps canonical plane points to image px
    n_frets: int = 24  # last fret drawn: voted / overridden, capped at the highest confident fret until settled
    kpts: np.ndarray = field(default_factory=lambda: np.full((N_KPTS, 3), np.nan, np.float32))  # raw x, y, conf
    board_score: float = 0.0
    anchors: list[Anchor] = field(default_factory=list)  # keypoints with conf > kpt_thr (canonical fret/side)
    end_estimate: int | None = None  # this frame's board-end evidence (None = none)
    held: bool = False  # True if `model` is the previous fit (no usable detection this frame)
    rmse_px: float = float("nan")  # fit residual of this frame's raw fit
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def H(self) -> np.ndarray | None:
        return None if self.model is None else self.model.H

    def wires(self) -> np.ndarray:
        """(n_frets + 1, 2, 2) fitted wire endpoints [bass, treble] x (x, y), wire 0 = nut. Empty if not valid."""
        if self.model is None:
            return np.zeros((0, 2, 2))
        return np.stack([self.model.fret_line(n) for n in range(self.n_frets + 1)])


def anchors_from(k: np.ndarray, thr: float = 0.5, max_fret: int = 24) -> list[Anchor]:
    """Anchors from one (50, 3) x, y, conf keypoint array (absolute slots 0b, 0t, ..., 24t)."""
    return [
        Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
        for i, (x, y, c) in enumerate(k)
        if np.isfinite(c) and c > thr and i // 2 <= max_fret
    ]


def safe_fit(anchors, n_frets: int = 24) -> FretboardModel | None:
    try:
        m = fit_fretboard(anchors, n_frets=n_frets)
    except (FitError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return m if np.isfinite(m.H).all() and np.isfinite(m.rmse_px) else None


def mirror_kpts(k: np.ndarray, width: int) -> np.ndarray:
    """Keypoints between canonical and original frame (x -> W - 1 - x, same convention as geometry.mirror_matrix)."""
    out = k.copy()
    out[:, 0] = width - 1 - out[:, 0]
    return out


class FretboardTracker:
    """Stateful per-video tracker. One instance per camera / clip; call `reset()` on a cut or a new guitar.

    model: path (.mlpackage / .onnx / .pt) or a Backend instance. mirrored: frame shows a right-handed guitar mirrored
    (or a lefty unmirrored), see D1. frets: known fret count (skips board-end detection). fps: nominal rate, only
    used when `process` gets no timestamps. smooth=False disables One-Euro. end_every=1 checks the board end on
    every frame (demo v2 behaviour).
    """

    def __init__(
        self,
        model: str | Backend,
        mirrored: bool = False,
        frets: int | None = None,
        fps: float = 30.0,
        smooth: bool = True,
        kpt_thr: float = 0.5,
        box_thr: float = 0.25,
        hold: int = 10,
        clip_kpts: bool = True,
        min_cutoff: float = 0.4,
        beta: float = 3.0,
        speed_floor: float = 2.0,
        end_every: int = 10,
        settled_votes: int = 60,
        **backend_kw,
    ):
        self.backend = model if isinstance(model, Backend) else load_backend(model, **backend_kw)
        self.mirrored, self.frets, self.fps, self.smooth = mirrored, frets, fps, smooth
        self.kpt_thr, self.box_thr, self.hold, self.clip_kpts = kpt_thr, box_thr, hold, clip_kpts
        # board-end check (probe fit + ridge test, ~half the post-processing time) runs every frame until the fret
        # count vote has `settled_votes` votes, then every `end_every`-th frame (the count of a guitar is constant)
        self.end_every, self.settled_votes = max(1, end_every), settled_votes
        self.one_euro = {"min_cutoff": min_cutoff, "beta": beta, "speed_floor": speed_floor}
        self._letterbox: Letterbox | None = None
        self.reset()

    def reset(self) -> None:
        self.vote = FretCountVote(override=self.frets)
        self.filter = OneEuroModel(self.fps, **self.one_euro) if self.smooth else None
        self.canon_model: FretboardModel | None = None
        self.missed = 0
        self.last_t: float | None = None
        self.frame_index = 0

    # -- stages (public so apps can run inference on another thread) ---------------------------------------------

    def canonical(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.flip(frame_bgr, 1) if self.mirrored else frame_bgr

    def infer(self, canon_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Canonical BGR frame -> ((50, 3) keypoints in canonical frame px, board score)."""
        if getattr(self.backend, "source_resolution", False):
            start = time.perf_counter()
            k, score = self.backend.predict_frame(canon_bgr, self.box_thr, self.clip_kpts)
            self.infer_ms = {"letterbox": 0.0, "model": (time.perf_counter() - start) * 1e3}
            return k, score
        lb = self._letterbox
        if lb is None or (lb.h, lb.w) != canon_bgr.shape[:2] or lb.size != self.backend.imgsz:
            lb = self._letterbox = Letterbox.for_shape(canon_bgr.shape, self.backend.imgsz)
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(lb.apply(canon_bgr), cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        raw = self.backend(rgb)
        t2 = time.perf_counter()
        k, score, _ = decode(raw, self.box_thr)
        if np.isfinite(k[:, 0]).all():
            k[:, :2] = lb.to_frame(k[:, :2], clip=self.clip_kpts)
        self.infer_ms = {"letterbox": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3}
        return k, score

    def update(self, canon_bgr: np.ndarray, k: np.ndarray, t: float | None = None) -> TrackState:
        """Post-process one frame's canonical keypoints (board end, vote, fit, smoothing, hold)."""
        dt = None
        if t is not None and self.last_t is not None:
            dt = t - self.last_t
        t = t if t is not None else (0.0 if self.last_t is None else self.last_t + 1.0 / self.fps)
        self.last_t = t

        anchors = anchors_from(k, self.kpt_thr)
        estimate = None
        settled = sum(self.vote.votes.values()) >= self.settled_votes
        self.frame_index += 1
        conf = np.nanmin(k[:, 2].reshape(25, 2), 1) if np.isfinite(k[:, 2]).any() else None
        if not self.frets and (not settled or self.frame_index % self.end_every == 0):
            probe = safe_fit([p for p in anchors if p.fret <= END_FIT_MAX_FRET])
            if probe is not None:
                estimate = frame_last_fret(ridge_scores(canon_bgr, probe), conf)
        self.vote.update(estimate)
        if conf is not None:
            self.vote.observe_conf(conf)
        n_frets = self.vote.shown  # capped at the highest confident fret until the vote settles

        fit = safe_fit([p for p in anchors if p.fret <= n_frets], n_frets)
        held = fit is None
        if fit is None:
            self.missed += 1
            if self.missed > self.hold:
                self.canon_model = None
                if self.filter:
                    self.filter.reset()
        else:
            self.missed = 0
            self.canon_model = self.filter.update(fit, canon_bgr.shape, dt) if self.filter else fit

        h, w = canon_bgr.shape[:2]
        model = self.canon_model
        if model is not None and self.mirrored:
            model = mirror_model(model, w)
        return TrackState(
            t=t,
            valid=model is not None,
            model=model,
            n_frets=n_frets,
            kpts=mirror_kpts(k, w) if self.mirrored else k,
            anchors=anchors,
            end_estimate=estimate,
            held=held and model is not None,
            rmse_px=fit.rmse_px if fit is not None else float("nan"),
        )

    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> TrackState:
        """Full pipeline on one BGR frame captured at time `t` (seconds; None = assume nominal fps)."""
        t0 = time.perf_counter()
        canon = self.canonical(frame_bgr)
        t1 = time.perf_counter()
        k, score = self.infer(canon)
        t2 = time.perf_counter()
        state = self.update(canon, k, t)
        t3 = time.perf_counter()
        state.board_score = score
        state.timings_ms = {
            "mirror": (t1 - t0) * 1e3,
            **self.infer_ms,
            "post": (t3 - t2) * 1e3,
            "total": (t3 - t0) * 1e3,
        }
        return state
