"""Board-end (last fret) detection for 21/22/24-fret guitars.

Per frame, the fitted model gives the predicted position of every wire, also beyond the last real one. We rectify the
high-fret strip of the (canonical) image through the fit and test each wire position for a fret-wire ridge: a thin
line brighter than the wood on *both* sides, across the strings. Pickups, the body edge and pole pieces give steps or
blobs, not two-sided ridges that span the board. The keypoint confidence of the high slots is combined with the ridge
score, and `FretCountVote` stabilizes the per-frame estimates over the clip (the fret count of a guitar is constant).
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from gro.geometry import FretboardModel, fret_u

FIRST_TESTED = 19  # every guitar we target has >= 20 frets; wires 15..18 serve as the reference ridge strength
REF_FRETS = range(14, 19)
PX_PER_CELL = 24  # rectified samples per fret cell at the high frets (upsampled where the image is coarser)


def rectify_strip(canon: np.ndarray, model: FretboardModel, f_lo: float, f_hi: float, rows: int = 24):
    """Gray strip sampled on canonical (fret, y) with y in [0.1, 0.9] (between the outer strings).

    Columns are uniform in fret number (not u), so every cell is PX_PER_CELL samples wide."""
    cols = int(round((f_hi - f_lo) * PX_PER_CELL))
    fr = np.linspace(f_lo, f_hi, cols)
    ys = np.linspace(0.1, 0.9, rows)
    F, Y = np.meshgrid(fr, ys)
    pts = model.project(fret_u(F.ravel()), Y.ravel()).reshape(rows, cols, 2).astype(np.float32)
    gray = canon if canon.ndim == 2 else cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    strip = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderValue=0)
    inside = (
        (pts[..., 0] >= 0) & (pts[..., 0] < gray.shape[1] - 1) & (pts[..., 1] >= 0) & (pts[..., 1] < gray.shape[0] - 1)
    )
    return strip.astype(np.float32), inside, fr


def ridge_scores(canon: np.ndarray, model: FretboardModel, n_max: int = 24) -> dict[int, float]:
    """Ridge strength at each wire REF_FRETS..n_max, normalized by the median strength of the reference wires.

    Missing keys = wire not (fully) in view. ~1 = looks like the reference wires, ~0 = no ridge."""
    f_lo, f_hi = REF_FRETS[0] - 0.5, n_max + 0.5
    strip, inside, fr = rectify_strip(canon, model, f_lo, f_hi)
    half = strip.shape[0] // 2
    d = PX_PER_CELL // 4  # flank distance: a quarter cell either side
    search = max(2, PX_PER_CELL // 6)  # tolerate fit error of ~1/6 cell
    raw = {}
    for n in range(REF_FRETS[0], n_max + 1):
        c = int(round((n - f_lo) * PX_PER_CELL))
        lo, hi = c - search - d, c + search + d
        if lo < 0 or hi >= strip.shape[1] or not inside[:, lo : hi + 1].all():
            continue
        best = -np.inf
        for x in range(c - search, c + search + 1):
            # two-sided ridge in each half of the strip (bass / treble): the wire must span the strings
            s = [min(p[x] - p[x - d], p[x] - p[x + d]) for p in (strip[:half].mean(0), strip[half:].mean(0))]
            best = max(best, min(s))
        raw[n] = best
    ref = [raw[n] for n in REF_FRETS if n in raw]
    if len(ref) < 3 or np.median(ref) <= 1.0:
        return {}
    r = float(np.median(ref))
    return {n: v / r for n, v in raw.items() if n >= FIRST_TESTED}


def frame_last_fret(ridge: dict[int, float], conf: np.ndarray | None = None, thr: float = 0.35) -> int | None:
    """Last fret supported in this frame, or None if the evidence is incomplete (high frets out of view).

    A wire counts when its ridge ratio >= thr, or >= thr/2 with confident keypoints on both sides (conf: (25,)
    min-over-sides confidence). Counting stops at the first wire without support; None if that is out of view."""
    n = FIRST_TESTED
    while n <= 24:
        if n not in ridge:
            return None
        c = conf[n] if conf is not None and np.isfinite(conf[n]) else 0.0
        if not (ridge[n] >= thr or (c > 0.5 and ridge[n] >= thr / 2)):
            break
        n += 1
    last = n - 1
    if last < 24 and last + 1 not in ridge:
        return None
    return last if last >= 20 else None  # fewer than 20 = occluded / bad fit, not a real guitar


class FretCountVote:
    """Per-clip fret count: running vote over per-frame estimates with hysteresis.

    Until `min_votes` frames have voted, `value` is the default (24, the model's slot count). After that it follows the
    mode, switching only when the new mode leads the current value by `margin` votes.

    `shown` is the count to draw. Until the vote has settled (min_votes votes and `value` is their mode) it is capped
    at the highest fret whose keypoints are confidently seen (`observe_conf`), so a 22-fret guitar does not get 24
    frets drawn for the first ~0.5 s. The cap is the highest per-frame confident fret seen in >= `cap_frames` frames
    (monotone, so the drawn board end only grows while the vote settles)."""

    def __init__(
        self,
        default: int = 24,
        min_votes: int = 15,
        margin: int = 10,
        override: int | None = None,
        cap_thr: float = 0.5,
        cap_frames: int = 3,
    ):
        self.votes: Counter[int] = Counter()
        self.value = override or default
        self.override = override
        self.min_votes, self.margin = min_votes, margin
        self.cap_thr, self.cap_frames = cap_thr, cap_frames
        self.caps: Counter[int] = Counter()

    def update(self, estimate: int | None) -> int:
        if self.override or estimate is None:
            return self.value
        self.votes[estimate] += 1
        if sum(self.votes.values()) >= self.min_votes:
            mode, count = self.votes.most_common(1)[0]
            if mode != self.value and count >= self.votes[self.value] + self.margin:
                self.value = mode
        return self.value

    @property
    def settled(self) -> bool:
        if self.override:
            return True
        n = sum(self.votes.values())
        return n >= self.min_votes and self.votes[self.value] == max(self.votes.values())

    def observe_conf(self, conf: np.ndarray) -> None:
        """Record one frame's (25,) min-over-sides keypoint confidence (only used until the vote settles)."""
        if self.settled:
            return
        c = np.nan_to_num(np.asarray(conf, float), nan=0.0)
        seen = np.flatnonzero(c > self.cap_thr)
        if len(seen) and seen[-1] >= 12:  # a usable view reaches at least the 12th fret
            self.caps[int(seen[-1])] += 1

    @property
    def cap(self) -> int | None:
        """Highest confidently seen fret (seen in >= cap_frames frames, or in every frame so far); None = no data."""
        if not self.caps:
            return None
        need = min(self.cap_frames, sum(self.caps.values()))
        total = 0
        for f in sorted(self.caps, reverse=True):
            total += self.caps[f]
            if total >= need:
                return f
        return None

    @property
    def shown(self) -> int:
        cap = None if self.settled else self.cap
        return self.value if cap is None else min(self.value, cap)
