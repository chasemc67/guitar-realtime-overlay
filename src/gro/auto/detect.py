"""Classical-CV neck localization and fret-wire detection (no learning).

Single-frame pipeline, all in the canonical (already mirrored if `video.mirrored`, D1) image:

1. **Wire candidates.** LSD line segments, kept if they are bright thin ridges (centre brighter than both flanks).
2. **Neck axis.** RANSAC over pairs of near-parallel, similar-length candidate segments: the axis runs through their
   midpoints and must cross them at a steep angle. Support = other candidates that are parallel-ish, similar length,
   centred on the axis. Fret wires are the only structure that forms a long such row (books and strings don't).
3. **Rectified strip.** The two board edges are fitted through the candidates' end points; the quad between the
   first and last supporting wire is warped to a rectangle, so wires become ~vertical. A column profile of the
   horizontal ridge response (median over the board rows, robust to fingers and inlays) gives wire peaks.
4. **1D fret grid.** Peak positions are fitted with a 1D projective map of the 12th-root-of-2 positions
   x_n = P(u_n) (RANSAC over triples of peaks and small index gaps, then least squares). Fret spacing can't fix the
   absolute index (PLAN D4), so indices here are *relative*; `gro.auto.numbering` resolves the shift.
5. **Anchors.** Each grid wire is mapped back to the image, snapped to its ridge (`gro.snap`), the board edges are
   re-fitted through the measured wire ends, and anchors = wire x edge intersections (same as the hand tool).
   A second pass re-rectifies with the fitted homography and re-checks every wire, so faint wires and wires
   partly under fingers are found (vis 1 when the ridge evidence is weak).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.strings import RHO_NOM, allowed_shifts, detect_strings, inset_lines, lines_rho, prior_lines, prior_ok
from gro.geometry import FitError, FretboardModel, apply_h, fit_fretboard, fret_u
from gro.snap import fit_line, intersect, robust_line, wire_extent

# ---------------------------------------------------------------------------------------------------------------
# Image helpers


def prep_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale + 3x3 Gaussian, the same preprocessing as the hand-anchor tool (scripts/hand_anchors.py)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.GaussianBlur(g, (3, 3), 0)


def sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples at (..., 2) points."""
    pts = np.asarray(pts, np.float32)
    shp = pts.shape[:-1]
    p = pts.reshape(-1, 2)
    n = len(p)
    cols = 4096  # remap maps must stay below SHRT_MAX per side
    rows = -(-n // cols)
    q = np.zeros((rows * cols, 2), np.float32)
    q[:n] = p
    q = q.reshape(rows, cols, 2)
    m = cv2.remap(gray, q[..., 0], q[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.ravel()[:n].reshape(shp).astype(float)


def ridge_profile_along(gray, p0, p1, t: np.ndarray, side_px: float = 3.0) -> np.ndarray:
    """Ridge response (centre minus mean of both flanks) at fractions `t` along p0->p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    c = p0 + t[:, None] * d
    return sample(gray, c) - 0.5 * (sample(gray, c + side_px * nrm) + sample(gray, c - side_px * nrm))


# ---------------------------------------------------------------------------------------------------------------
# 1. Wire candidates


@dataclass
class Segments:
    p0: np.ndarray  # (N, 2)
    p1: np.ndarray
    ridge: np.ndarray  # best ridge score (gray levels)
    offset: np.ndarray  # across-segment offset of the ridge centre

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    @property
    def length(self) -> np.ndarray:
        return np.linalg.norm(self.p1 - self.p0, axis=1)

    @property
    def angle(self) -> np.ndarray:
        """Direction angle in [0, pi)."""
        d = self.p1 - self.p0
        return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)

    def __len__(self) -> int:
        return len(self.p0)

    def subset(self, idx) -> Segments:
        return Segments(self.p0[idx], self.p1[idx], self.ridge[idx], self.offset[idx])


def wire_candidates(gray: np.ndarray, min_len: float = 12.0, min_ridge: float = 6.0) -> Segments:
    """LSD segments that sit on (or right next to) a bright thin ridge. Segments are re-centred on the ridge."""
    lsd = cv2.createLineSegmentDetector()
    raw = lsd.detect(gray)[0]
    if raw is None:
        return Segments(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    raw = raw.reshape(-1, 4).astype(float)
    p0, p1 = raw[:, :2], raw[:, 2:]
    L = np.linalg.norm(p1 - p0, axis=1)
    keep = (min_len <= L) & (L <= 0.6 * min(gray.shape))
    p0, p1, L = p0[keep], p1[keep], L[keep]
    if len(p0) == 0:
        return Segments(p0, p1, np.zeros(0), np.zeros(0))
    d = (p1 - p0) / L[:, None]
    nrm = np.c_[-d[:, 1], d[:, 0]]
    t = np.linspace(0.15, 0.85, 15)
    offs = np.arange(-3.0, 3.01, 0.5)
    c = p0[:, None, None, :] + t[None, None, :, None] * (p1 - p0)[:, None, None, :]  # N,1,T,2
    c = c + offs[None, :, None, None] * nrm[:, None, None, :]  # N,O,T,2
    side = 3.0 * nrm[:, None, None, :]
    resp = sample(gray, c) - 0.5 * (sample(gray, c + side) + sample(gray, c - side))
    med = np.median(resp, axis=2)  # N,O
    k = np.argmax(med, axis=1)
    best = med[np.arange(len(k)), k]
    o = offs[k]
    sel = best >= min_ridge
    shift = (o[:, None] * nrm)[sel]
    return Segments(p0[sel] + shift, p1[sel] + shift, best[sel], o[sel])


# ---------------------------------------------------------------------------------------------------------------
# 2. Neck axis hypotheses


@dataclass
class NeckHypothesis:
    center: np.ndarray  # point on the axis
    axis: np.ndarray  # unit direction along the neck
    members: np.ndarray  # indices into the candidate Segments
    score: float


def _angle_diff(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def _unique_positions(t: np.ndarray, tol: float = 2.5) -> int:
    if len(t) == 0:
        return 0
    s = np.sort(t)
    return int(1 + np.sum(np.diff(s) > tol))


def neck_hypotheses(
    segs: Segments,
    max_pairs: int = 4000,
    max_angle_deg: float = 14.0,
    top_k: int = 3,
    seed: int = 0,
) -> list[NeckHypothesis]:
    n = len(segs)
    if n < 4:
        return []
    mid, L, ang = segs.mid, segs.length, segs.angle
    i, j = np.triu_indices(n, 1)
    dm = mid[j] - mid[i]
    sep = np.linalg.norm(dm, axis=1)
    lm = 0.5 * (L[i] + L[j])
    ok = _angle_diff(ang[i], ang[j]) < np.deg2rad(8)
    ok &= (np.minimum(L[i], L[j]) / np.maximum(L[i], L[j])) > 0.5
    ok &= (sep > 0.12 * lm) & (sep < 4.0 * lm)
    seg_dir = np.c_[np.cos(ang[i]), np.sin(ang[i])]
    dmn = dm / np.maximum(sep, 1e-9)[:, None]
    cross = np.abs(seg_dir[:, 0] * dmn[:, 1] - seg_dir[:, 1] * dmn[:, 0])
    ok &= cross > 0.75
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(i) > max_pairs:
        pick = rng.choice(len(i), max_pairs, replace=False)
        i, j = i[pick], j[pick]
    c = 0.5 * (mid[i] + mid[j])  # P,2
    a = mid[j] - mid[i]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    nrm = np.c_[-a[:, 1], a[:, 0]]
    th = ang[i]
    lm = 0.5 * (L[i] + L[j])
    rel = mid[None, :, :] - c[:, None, :]  # P,N,2
    off = np.abs(np.einsum("pnk,pk->pn", rel, nrm))
    along = np.einsum("pnk,pk->pn", rel, a)
    sd = np.stack([np.cos(ang), np.sin(ang)], 1)  # N,2
    crossk = np.abs(a[:, 0:1] * sd[None, :, 1] - a[:, 1:2] * sd[None, :, 0])
    sup = _angle_diff(ang[None, :], th[:, None]) < np.deg2rad(max_angle_deg)
    sup &= off < 0.3 * np.maximum(L[None, :], lm[:, None])
    r = L[None, :] / lm[:, None]
    sup &= (r > 0.3) & (r < 3.0)
    sup &= crossk > 0.7
    # Count distinct wires (the two LSD edges of one wire collapse after ridge re-centring) and weight by ridge.
    scores = np.zeros(len(i))
    for p in range(len(i)):
        idx = np.flatnonzero(sup[p])
        scores[p] = _unique_positions(along[p, idx])
    order = np.argsort(-scores)
    hyps: list[NeckHypothesis] = []
    for p in order:
        if scores[p] < 5 or len(hyps) >= top_k:
            break
        members = np.flatnonzero(sup[p])
        if any(len(np.intersect1d(members, h.members)) > 0.4 * min(len(members), len(h.members)) for h in hyps):
            continue
        # Re-fit the axis through the member midpoints.
        cc, aa = robust_line(mid[members])[0]
        hyps.append(NeckHypothesis(cc, aa / np.linalg.norm(aa), members, float(scores[p])))
    return hyps


# ---------------------------------------------------------------------------------------------------------------
# 3. Rectified strip


@dataclass
class Strip:
    """Neck-aligned resampling. Column x <-> axis position t = t0 + x; each column samples the image along the local
    wire direction (towards the wires' vanishing point), so wires are exactly vertical when the vanishing point is
    right, and a wire's column is a 1D projective function of its u (any image line is the image of a board line)."""

    center: np.ndarray
    axis: np.ndarray
    vp: np.ndarray  # homogeneous vanishing point of the wires
    t0: float
    width: int
    len_fit: tuple[float, float]  # local wire length L(t) = a + b t
    rows: int = 48
    half: float = 0.8  # rows span s in [-half, half] x L(t)
    image: np.ndarray | None = None
    down: np.ndarray | None = None  # across-neck unit vector pointing image-down (to the treble edge)

    def length(self, t):
        a, b = self.len_fit
        return np.maximum(a + b * np.asarray(t, float), 4.0)

    def wire_dir(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        p = self.center + t[..., None] * self.axis
        v = self.vp[:2] - self.vp[2] * p  # direction from p towards the vanishing point (also fine at infinity)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        sgn = np.sign(v @ self.down)
        sgn[sgn == 0] = 1
        return v * sgn[..., None]

    def point(self, x, s) -> np.ndarray:
        """Image point for strip column x and normalized across-neck coordinate s (0 = axis, +-0.5 ~ board edges)."""
        x, s = np.broadcast_arrays(np.asarray(x, float), np.asarray(s, float))
        t = self.t0 + x
        return self.center + t[..., None] * self.axis + (s * self.length(t))[..., None] * self.wire_dir(t)

    def x_of(self, pts) -> np.ndarray:
        return (np.asarray(pts, float) - self.center) @ self.axis - self.t0

    def row_s(self, j) -> np.ndarray:
        return (np.asarray(j, float) / (self.rows - 1) * 2 - 1) * self.half


def vanishing_point(p0: np.ndarray, p1: np.ndarray, w: np.ndarray | None = None, iters: int = 5) -> np.ndarray:
    """Least-squares common point of lines p0-p1 (homogeneous, unit norm), with trimming of the worst lines."""
    scale = max(np.abs(np.r_[p0.ravel(), p1.ravel()]).max(), 1.0)
    a = np.c_[p0 / scale, np.ones(len(p0))]
    b = np.c_[p1 / scale, np.ones(len(p1))]
    lines = np.cross(a, b)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    w = np.ones(len(lines)) if w is None else np.asarray(w, float)
    keep = np.ones(len(lines), bool)
    v = None
    for _ in range(iters):
        _, _, vt = np.linalg.svd(lines[keep] * w[keep, None])
        v = vt[-1]
        # Distance of each line to the point, in image px (angle-ish when v is at infinity).
        d = np.abs(lines @ v) / max(np.linalg.norm(v), 1e-12)
        thr = max(3 * np.median(d[keep]), 1e-6)
        new = d <= thr
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    v = v * np.array([scale, scale, 1.0])
    return v / np.linalg.norm(v)


def make_strip(gray: np.ndarray, segs: Segments, hyp: NeckHypothesis, pad: float = 0.6, rows: int = 48) -> Strip | None:
    m = hyp.members
    if len(m) < 4:
        return None
    a = hyp.axis
    down = np.array([-a[1], a[0]])
    if down[1] < 0:
        down = -down
    # Axis through the member midpoints; t along it.
    t_mid = (segs.mid[m] - hyp.center) @ a
    L = segs.length[m]
    vp = vanishing_point(segs.p0[m], segs.p1[m], w=L)
    # Local wire length: robust line through (t, L), keeping the longer (complete) segments.
    keep = np.ones(len(m), bool)
    coef = (float(np.median(L)), 0.0)
    for _ in range(4):
        if keep.sum() < 3:
            break
        A_ = np.c_[np.ones(keep.sum()), t_mid[keep]]
        coef = tuple(np.linalg.lstsq(A_, L[keep], rcond=None)[0])
        fit = coef[0] + coef[1] * t_mid
        keep = L >= 0.85 * fit
    span = t_mid.max() - t_mid.min()
    if span < 20:
        return None
    t0 = t_mid.min() - pad * span
    width = int(span * (1 + 2 * pad))
    st = Strip(hyp.center, a, vp, t0, width, coef, rows=rows, down=down)
    xs = np.arange(width, dtype=float)
    ss = st.row_s(np.arange(rows))
    X, S = np.meshgrid(xs, ss)
    pts = st.point(X, S).astype(np.float32)
    h, w = gray.shape[:2]
    inside = (pts[..., 0] >= 0) & (pts[..., 0] < w - 1) & (pts[..., 1] >= 0) & (pts[..., 1] < h - 1)
    img = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    st.image = img
    st.inside = inside
    return st


def column_profile(strip: Strip, side_px: float = 2.5) -> np.ndarray:
    """Per-column ridge response for vertical wires over the inner board rows. A high percentile of the rows is used
    (wires are bright along their whole length; fingers and inlays only cover part of it)."""
    g = strip.image.astype(np.float32)
    s = strip.row_s(np.arange(strip.rows))
    sel = np.abs(s) <= 0.38
    rows = g[sel]
    k = int(round(side_px))
    resp = rows[:, k:-k] - 0.5 * (rows[:, : -2 * k] + rows[:, 2 * k :])
    prof = np.zeros(g.shape[1])
    # 60th percentile: robust to fingers covering up to ~40% of the rows.
    prof[k:-k] = np.percentile(resp, 60, axis=0)
    valid = strip.inside[sel].mean(axis=0) > 0.95
    prof[~valid] = 0.0
    return np.convolve(prof, [0.25, 0.5, 0.25], mode="same")


def find_peaks(prof: np.ndarray, min_height: float, min_dist: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima above `min_height` (sub-pixel by parabola), greedily separated by `min_dist`."""
    p = prof
    cand = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > min_height)) + 1
    cand = cand[np.argsort(-p[cand])]
    taken: list[int] = []
    for c in cand:
        if all(abs(c - t) >= min_dist for t in taken):
            taken.append(int(c))
    taken.sort()
    xs, hs = [], []
    for c in taken:
        a, b, d = p[c - 1], p[c], p[c + 1]
        den = a - 2 * b + d
        off = 0.5 * (a - d) / den if abs(den) > 1e-9 else 0.0
        xs.append(c + float(np.clip(off, -0.5, 0.5)))
        hs.append(b)
    return np.array(xs), np.array(hs)


# ---------------------------------------------------------------------------------------------------------------
# 4. 1D fret grid: x_n = (a u_n + b) / (c u_n + 1), u_n = 1 - 2^(-n/12)


@dataclass
class Grid1D:
    abc: np.ndarray
    sign: int  # +1: index increases with x, -1: decreases
    peak_idx: dict[int, int]  # relative fret index -> peak number (inliers)
    rmse: float
    score: float

    def x(self, n) -> np.ndarray:
        u = fret_u(np.asarray(n, float))
        a, b, c = self.abc
        return (a * u + b) / (c * u + 1.0)

    @property
    def index_range(self) -> tuple[int, int]:
        k = sorted(self.peak_idx)
        return k[0], k[-1]


def _solve_abc(u: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray | None:
    A_ = np.c_[u, np.ones_like(u), -u * x]
    if w is not None:
        A_, x = A_ * w[:, None], x * w
    try:
        sol, *_ = np.linalg.lstsq(A_, x, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return sol


def _grid_eval(abc, xs, hs, n_lo, n_hi, x_range, tol_frac):
    n = np.arange(n_lo, n_hi + 1)
    u = fret_u(n)
    den = abc[2] * u + 1.0
    if np.any(den <= 1e-6):
        return None
    xp = (abc[0] * u + abc[1]) / den
    dx = np.diff(xp)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        return None
    spacing = np.abs(np.r_[dx, dx[-1]])
    # Assign each peak to its nearest predicted wire.
    k = np.clip(np.searchsorted(xp, xs) if dx[0] > 0 else np.searchsorted(-xp, -xs), 1, len(xp) - 1)
    near = np.where(np.abs(xp[k - 1] - xs) < np.abs(xp[k] - xs), k - 1, k)
    err = np.abs(xp[near] - xs)
    tol = np.maximum(tol_frac * spacing[near], 1.2)
    inl = err < tol
    # One peak per wire: keep the strongest.
    best: dict[int, int] = {}
    for pi in np.flatnonzero(inl):
        w = int(n[near[pi]])
        if w not in best or hs[pi] > hs[best[w]]:
            best[w] = int(pi)
    if len(best) < 3:
        return None
    # Missing wires between the first and last matched one (penalty against aliasing to a denser grid).
    lo, hi = min(best), max(best)
    # Direction prior: physical fret spacing shrinks 3.8x from fret 1 to 24, far more than any close-up
    # perspective, so the image spacing must shrink with the index. (A reversed geometric sequence is itself a 1D
    # projective image of the forward one, x -> 1/x, so the fit alone can't tell the direction.)
    i_lo, i_hi = np.flatnonzero(n == lo)[0], np.flatnonzero(n == hi)[0]
    if spacing[i_lo] <= spacing[i_hi]:
        return None
    inside = (n >= lo) & (n <= hi) & (xp >= x_range[0]) & (xp <= x_range[1])
    missing = int(inside.sum()) - len(best)
    e = np.array(
        [
            abs(xp[np.flatnonzero(n == w)[0]] - xs[p]) / max(spacing[np.flatnonzero(n == w)[0]], 1e-6)
            for w, p in best.items()
        ]
    )
    score = len(best) - 0.5 * missing - 2.0 * float(np.mean(e))
    return best, score, float(np.sqrt(np.mean(e**2)))


def fit_grid_1d(
    xs: np.ndarray,
    hs: np.ndarray,
    x_range: tuple[float, float],
    iters: int = 1500,
    tol_frac: float = 0.15,
    seed: int = 0,
) -> Grid1D | None:
    """RANSAC: 3 peaks + small index gaps -> (a, b, c); keep the best-scoring grid, then refit on its inliers."""
    m = len(xs)
    if m < 4:
        return None
    rng = np.random.default_rng(seed)
    best = None
    n_lo, n_hi = -40, 40
    for _ in range(iters):
        i = int(rng.integers(0, m - 2))
        j = int(min(m - 2, i + rng.integers(1, 3)))
        k = int(min(m - 1, j + rng.integers(1, 4)))
        if not i < j < k:
            continue
        gj, gk = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        for sign in (1, -1):
            n3 = np.array([0, gj, gj + gk]) * sign
            abc = _solve_abc(fret_u(n3.astype(float)), xs[[i, j, k]])
            if abc is None:
                continue
            ev = _grid_eval(abc, xs, hs, n_lo, n_hi, x_range, tol_frac)
            if ev is None:
                continue
            if best is None or ev[1] > best[2]:
                best = (abc, sign, ev[1], ev[0])
    if best is None:
        return None
    abc, sign, score, assign = best
    for _ in range(5):
        ns = np.array(sorted(assign), float)
        pk = np.array([assign[int(n)] for n in ns])
        sol = _solve_abc(fret_u(ns), xs[pk], np.sqrt(np.maximum(hs[pk], 1e-3)))
        if sol is None:
            break
        ev = _grid_eval(sol, xs, hs, n_lo, n_hi, x_range, tol_frac)
        if ev is None:
            break
        abc, (assign, score, rmse) = sol, ev
    ev = _grid_eval(abc, xs, hs, n_lo, n_hi, x_range, tol_frac)
    if ev is None:
        return None
    assign, score, rmse = ev
    # Normalise relative indices so the first matched wire is 1 (nut side = smaller index).
    lo = min(assign)
    abc2 = _reindex(abc, lo - 1)
    return Grid1D(abc2, sign, {n - (lo - 1): p for n, p in assign.items()}, rmse, score)


def _reindex(abc: np.ndarray, s: int) -> np.ndarray:
    """Coefficients for n' = n - s: u_{n'+s} = 1 - 2^(-s/12) (1 - u_{n'}) = alpha u' + beta (affine)."""
    k = 2.0 ** (-s / 12.0)
    alpha, beta = k, 1.0 - k
    a, b, c = abc
    # x = (a (alpha u' + beta) + b) / (c (alpha u' + beta) + 1); normalise the constant term to 1.
    num_u, num_1 = a * alpha, a * beta + b
    den_u, den_1 = c * alpha, c * beta + 1.0
    return np.array([num_u / den_1, num_1 / den_1, den_u / den_1])


# ---------------------------------------------------------------------------------------------------------------
# 5. Wires -> anchors


@dataclass
class WireObs:
    fret: int  # relative index
    seg: np.ndarray  # 2x2 snapped (or predicted) wire, bass end first
    score: float  # median ridge response along the wire
    support: float  # fraction of samples along the wire with a clear ridge
    vis: int


@dataclass
class Detection:
    ok: bool
    reason: str = ""
    anchors: list[A.Anchor] = field(default_factory=list)  # relative fret indices
    model: FretboardModel | None = None
    wires: dict[int, WireObs] = field(default_factory=dict)
    edges: dict | None = None
    grid: Grid1D | None = None
    strip: Strip | None = None
    hyp: NeckHypothesis | None = None
    n_candidates: int = 0
    debug: dict = field(default_factory=dict)

    @property
    def rel_range(self) -> tuple[int, int]:
        f = [w.fret for w in self.wires.values()]
        return (min(f), max(f)) if f else (0, -1)


def wire_support(gray, b, t, side_px: float = 3.0, level: float = 6.0) -> tuple[float, float]:
    """(median ridge response, fraction of samples above `level`) over the inner 80% of the wire b->t."""
    r = ridge_profile_along(gray, b, t, np.linspace(0.1, 0.9, 41), side_px)
    return float(np.median(r)), float(np.mean(r > level))


def _bass_first(seg: np.ndarray, nrm_down: np.ndarray) -> np.ndarray:
    return seg if (seg[0] @ nrm_down) <= (seg[1] @ nrm_down) else seg[::-1].copy()


def ransac_line(points, tol: float = 1.5, iters: int = 300, seed: int = 0):
    """Max-consensus line (|dist| < tol), then TLS on the inliers. Returns ((point, unit_dir), inlier mask)."""
    p = np.asarray(points, float)
    n = len(p)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters if n > 2 else 0):
        i, j = rng.choice(n, 2, replace=False)
        d = p[j] - p[i]
        if np.linalg.norm(d) < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        dist = np.abs((p - p[i]) @ nrm)
        inl = dist < tol
        sc = -float(np.sum(np.minimum(dist / tol, 1.0) ** 2))  # MSAC
        if best is None or sc > best[0]:
            best = (sc, inl)
    keep = best[1] if best is not None and best[1].sum() >= 2 else np.ones(n, bool)
    for _ in range(3):
        c, u = fit_line(p[keep])
        nrm = np.array([-u[1], u[0]])
        new = np.abs((p - c) @ nrm) < tol
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return fit_line(p[keep]), keep


def _line_hits(P1, P2, mids, dirs):
    """Distance along each wire (from its midpoint, along `dirs`) to the line(s) P1-P2. P1, P2: (C, 2)."""
    e = P2 - P1  # C,2
    # Solve mid + d * dir = P1 + s * e  ->  d = cross(P1 - mid, e) / cross(dir, e)
    rel = P1[:, None, :] - mids[None, :, :]  # C,K,2
    num = rel[..., 0] * e[:, None, 1] - rel[..., 1] * e[:, None, 0]
    den = dirs[None, :, 0] * e[:, None, 1] - dirs[None, :, 1] * e[:, None, 0]
    return num / np.where(np.abs(den) < 1e-9, 1e-9, den)


def fit_edge_line(
    gray: np.ndarray,
    segs: list[np.ndarray],
    side: int,
    init: tuple | None = None,
    search: float = 14.0,
    side_px: float = 3.0,
    clip: float = 4.0,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
    """Board edge where the wires end, fitted jointly over all wires.

    For each wire (2x2, bass end first) the ridge response r(d) is sampled from the wire's midpoint outwards
    (`side` 0 = bass end, 1 = treble end). With tau = half the wire's inner ridge level, the step score
    C(d) = cumsum((r - tau) / tau) peaks where the wire ends; per-wire 'first drop' ends are noisy where the board
    edge has no contrast (dark board on a dark shirt) or the wire tip is dim. The edge line maximises
    sum_k clip(C_k(d_k) - max C_k, -clip), searched over the line's hit distance on the first and last wire.
    Returns ((point, unit_dir), per-wire end distances) or None.
    """
    if len(segs) < 3:
        return None
    segs = [np.asarray(s_, float) for s_ in segs]
    mids = np.array([0.5 * (s_[0] + s_[1]) for s_ in segs])
    ends = np.array([s_[side] for s_ in segs])
    L = np.linalg.norm(ends - mids, axis=1)  # half lengths
    dirs = (ends - mids) / L[:, None]
    step = 0.25
    curves = []
    for k in range(len(segs)):
        d = np.arange(0.0, 1.8 * L[k], step)
        pts = mids[k] + d[:, None] * dirs[k]
        nrm = np.array([-dirs[k][1], dirs[k][0]])
        r = sample(gray, pts) - 0.5 * (sample(gray, pts + side_px * nrm) + sample(gray, pts - side_px * nrm))
        inner = r[(d > 0.1 * L[k]) & (d < 0.6 * L[k])]
        tau = max(0.5 * float(np.median(inner)) if len(inner) else 1.0, 2.0)
        C = np.cumsum((r - tau) / tau) * step
        curves.append((d, C - C.max()))
    if init is None:
        e0 = np.array([L[0], L[-1]])
    else:
        e0 = np.asarray(init, float)
    best = None
    for rad, st in ((search, 0.5), (1.0, 0.1)):
        g = np.arange(-rad, rad + 1e-9, st)
        A_, B_ = np.meshgrid(g, g, indexing="ij")
        ea, eb = e0[0] + A_.ravel(), e0[1] + B_.ravel()
        P1 = mids[0] + ea[:, None] * dirs[0]
        P2 = mids[-1] + eb[:, None] * dirs[-1]
        dk = _line_hits(P1, P2, mids, dirs)  # C,K
        score = np.zeros(len(ea))
        for k, (d, C) in enumerate(curves):
            v = np.interp(dk[:, k], d, C, left=-1e3, right=-1e3)
            score += np.maximum(v, -clip)
        i = int(np.argmax(score))
        e0 = np.array([ea[i], eb[i]])
        best = (P1[i], P2[i], dk[i], score[i])
    P1, P2, dk, _ = best
    u = (P2 - P1) / max(np.linalg.norm(P2 - P1), 1e-9)
    return (P1, u), dk


def cell_lines(gray, segs: list[np.ndarray], side: int, max_lines: int = 6, min_resp: float = 6.0, seed: int = 0):
    """Long straight bright lines crossing the fret cells on one side of the neck (strings, the board-edge highlight):
    thin-ridge peaks sampled down the middle of each cell, grouped by RANSAC lines. Returns a list of
    (support fraction of cells, point, unit_dir, outward distance from the board centre)."""
    pts, cell = [], []
    mids = []
    for k in range(len(segs) - 1):
        a, b = segs[k], segs[k + 1]
        m = 0.25 * (a[0] + a[1] + b[0] + b[1])
        e = 0.5 * (a[side] + b[side])
        L = np.linalg.norm(e - m)
        if L < 4:
            continue
        d0 = (e - m) / L
        d = np.arange(0, 1.5 * L, 0.25)
        v = sample(gray, m + d[:, None] * d0)
        r = v[8:-8] - 0.5 * (v[:-16] + v[16:])  # +-2 px
        dd = d[8:-8]
        mids.append(m)
        for i in range(1, len(r) - 1):
            if r[i] > r[i - 1] and r[i] >= r[i + 1] and r[i] > min_resp:
                pts.append(m + dd[i] * d0)
                cell.append(k)
    if len(pts) < 3:
        return []
    pts, cell = np.array(pts), np.array(cell)
    ncell = len(set(cell.tolist()))
    centre = np.mean(mids, axis=0)
    rng = np.random.default_rng(seed)
    rem = np.ones(len(pts), bool)
    out = []
    for _ in range(max_lines):
        idx = np.flatnonzero(rem)
        if len(idx) < 3:
            break
        best = None
        for _ in range(300):
            i, j = rng.choice(idx, 2, replace=False)
            if cell[i] == cell[j]:
                continue
            dv = pts[j] - pts[i]
            nv = np.array([-dv[1], dv[0]]) / np.linalg.norm(dv)
            inl = idx[np.abs((pts[idx] - pts[i]) @ nv) < 1.2]
            sc = len(set(cell[inl].tolist()))
            if best is None or sc > best[0]:
                best = (sc, inl)
        if best is None or best[0] < 3:
            break
        c, u = fit_line(pts[best[1]])
        nv = np.array([-u[1], u[0]])
        rem &= ~(np.abs((pts - c) @ nv) < 2.0)
        sd = segs[0][side] - 0.5 * (segs[0][0] + segs[0][1])
        sgn = 1.0 if nv @ sd > 0 else -1.0
        out.append((best[0] / ncell, c, u, float(sgn * ((c - centre) @ nv))))
    return out


def _clamp_edge(edge, lines, segs, side, min_support: float = 0.6, overshoot_px: float = 4.0):
    """If the wire-end edge lies outside the outermost well-supported long line (board-edge highlight / outer
    string), the extra wire length is the visible side of the neck (fret tangs): use that line instead."""
    strong = [ln for ln in lines if ln[0] >= min_support]
    if not strong:
        return edge, False
    _, c, u, _ = max(strong, key=lambda ln: ln[3])
    hits_e, hits_l = [], []
    for sg in segs:
        m = 0.5 * (sg[0] + sg[1])
        d = (sg[side] - m) / np.linalg.norm(sg[side] - m)
        try:
            hits_e.append(np.dot(intersect((m, d), edge) - m, d))
            hits_l.append(np.dot(intersect((m, d), (c, u)) - m, d))
        except np.linalg.LinAlgError:
            continue
    if hits_e and np.median(np.subtract(hits_e, hits_l)) > overshoot_px:
        return (c, u), True
    return edge, False


def anchors_from_segments(
    gray: np.ndarray,
    segs: dict[int, tuple[np.ndarray, int]],
    nrm_down: np.ndarray,
    edge_lines=None,
    edge_tol: float = 2.5,
    refine_edges: bool = True,
) -> tuple[list[A.Anchor], dict]:
    """Board edges = RANSAC lines through the measured wire-ridge ends (`wire_extent`) of the visible wires; anchors =
    wire line x edge line. Wires whose two ends are both far off the edges are not fret wires (pickup poles, bridge,
    headstock text) and are dropped (returned in info["rejected"]). `edge_lines` (bass, treble) overrides the edges."""
    ends: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f, (seg, vis) in segs.items():
        if vis != 2 or np.linalg.norm(seg[1] - seg[0]) < 8:
            continue
        ext = wire_extent(gray, seg[0], seg[1])
        if ext is None:
            continue
        e0, e1 = np.asarray(ext[0]), np.asarray(ext[1])
        if e0 @ nrm_down > e1 @ nrm_down:
            e0, e1 = e1, e0
        ends[f] = (e0, e1)
    info: dict = {"rejected": []}
    if edge_lines is None:
        if len(ends) < 3:
            raise FitError("too few wire ends for the board edges")
        fr = list(ends)
        pb = np.array([ends[f][0] for f in fr])
        pt = np.array([ends[f][1] for f in fr])
        (be, kb), (te, kt) = ransac_line(pb, edge_tol), ransac_line(pt, edge_tol)
        info["edges"] = {"bass": (be, pb, kb), "treble": (te, pt, kt)}
        bad = {f for f, a, b in zip(fr, kb, kt, strict=True) if not a and not b}
        # Only drop off-edge wires at the ends of the run (inner wires may just have a finger over an end).
        inner = [f for f in fr if f not in bad]
        if inner:
            lo, hi = min(inner), max(inner)
            info["rejected"] = sorted(f for f in bad if f < lo or f > hi)
        if refine_edges:
            keep = [f for f in sorted(segs) if segs[f][1] == 2 and f not in info["rejected"]]
            ksegs = [segs[f][0] for f in keep]
            lines = []
            for side, ln in ((0, be), (1, te)):
                m0 = 0.5 * (ksegs[0][0] + ksegs[0][1])
                m1 = 0.5 * (ksegs[-1][0] + ksegs[-1][1])
                d0 = (ksegs[0][side] - m0) / np.linalg.norm(ksegs[0][side] - m0)
                d1 = (ksegs[-1][side] - m1) / np.linalg.norm(ksegs[-1][side] - m1)
                init = (
                    float(np.dot(intersect((m0, d0), ln) - m0, d0)),
                    float(np.dot(intersect((m1, d1), ln) - m1, d1)),
                )
                res = fit_edge_line(gray, ksegs, side, init=init)
                edge = ln if res is None else res[0]
                edge, clamped = _clamp_edge(edge, cell_lines(gray, ksegs, side), ksegs, side)
                info[f"clamped_{'bass' if side == 0 else 'treble'}"] = clamped
                lines.append(edge)
            be, te = lines
            info["edges_refined"] = True
    else:
        be, te = edge_lines
        info["edges"] = {}
    out = []
    for fret, (seg, vis) in sorted(segs.items()):
        if fret in info["rejected"]:
            continue
        wl = (seg[0], (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0]))
        b, t = intersect(wl, be), intersect(wl, te)
        out.append(A.Anchor(fret, "bass", float(b[0]), float(b[1]), vis))
        out.append(A.Anchor(fret, "treble", float(t[0]), float(t[1]), vis))
    info["lines"] = (be, te)
    return out, info


def snap_wire_fast(
    gray: np.ndarray,
    p0,
    p1,
    search: float = 4.0,
    step: float = 0.5,
    t0: float = 0.15,
    t1: float = 0.85,
    n: int = 40,
    side_px: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Vectorized `gro.snap.snap_wire` (same objective and tie-breaking, one remap call instead of ~900)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    offs = np.arange(-search, search + 1e-9, step)
    A_, B_ = np.meshgrid(offs, offs, indexing="ij")
    q0 = p0 + A_.ravel()[:, None] * nrm  # K,2
    q1 = p1 + B_.ravel()[:, None] * nrm
    dq = q1 - q0
    nq = np.c_[-dq[:, 1], dq[:, 0]] / np.maximum(np.linalg.norm(dq, axis=1), 1e-9)[:, None]
    t = np.linspace(t0, t1, n)
    c = q0[:, None, :] + t[None, :, None] * dq[:, None, :]  # K,n,2
    side = side_px * nq[:, None, :]
    resp = sample(gray, c) - 0.5 * (sample(gray, c + side) + sample(gray, c - side))
    score = np.median(resp, axis=1)
    k = int(np.argmax(score))
    return np.array([q0[k], q1[k]]), float(score[k])


@dataclass
class DetectParams:
    min_seg_len: float = 12.0
    min_ridge: float = 6.0
    snap_search: float = 4.0
    vis_support: float = 0.5  # fraction of the wire with a ridge to count as visible (else vis 1)
    min_wire_score: float = 4.0
    extend_support: float = 0.6  # beyond the grid range, a wire must be this clear to be added
    n_hyps: int = 6
    n_full: int = 3  # hypotheses (best grid score first) that get the full wire stage
    seed: int = 0


def detect_frame(img: np.ndarray, params: DetectParams | None = None, roi: np.ndarray | None = None) -> Detection:
    """Detect the fret grid in a canonical-orientation BGR frame. `roi` (uint8 mask) restricts the candidates."""
    P = params or DetectParams()
    gray = prep_gray(img)
    segs = wire_candidates(gray, P.min_seg_len, P.min_ridge)
    if roi is not None and len(segs):
        m = segs.mid.astype(int)
        h, w = roi.shape
        inside = roi[np.clip(m[:, 1], 0, h - 1), np.clip(m[:, 0], 0, w - 1)] > 0
        segs = segs.subset(np.flatnonzero(inside))
    det = Detection(False, n_candidates=len(segs))
    hyps = neck_hypotheses(segs, top_k=P.n_hyps, seed=P.seed)
    if not hyps:
        det.reason = "no neck hypothesis (too few parallel wire-like segments)"
        return det
    staged = []
    for hyp in hyps:
        d = _grid_stage(gray, segs, hyp, P)
        d.n_candidates = len(segs)
        staged.append(d)
    with_grid = sorted((d for d in staged if d.grid is not None), key=lambda d: -d.grid.score)
    if not with_grid:
        return staged[0]
    best: Detection | None = None
    for d in with_grid[: P.n_full]:
        _wire_stage(gray, d, P)
        if d.ok and (best is None or _quality(d) > _quality(best)):
            best = d
    return best or with_grid[0]


def _quality(d: Detection) -> float:
    """Visible wires + grid score - fit error, plus a neck-coherence bonus: real fret wires lie under six strings
    (a comb within the width prior), overlay text / burned-in logos (vertical letter strokes) don't."""
    n_vis = sum(w.vis == 2 for w in d.wires.values())
    neck = 8.0 * min(getattr(d.debug.get("strings"), "score", 0.0) or 0.0, 1.0)
    if d.debug.get("strings_source") == "comb":
        neck += 6.0
    return n_vis + 0.5 * d.grid.score - 2.0 * (d.model.rmse_px if d.model else 10) + neck


def _grid_stage(gray, segs: Segments, hyp: NeckHypothesis, P: DetectParams) -> Detection:
    det = Detection(False, hyp=hyp)
    strip = make_strip(gray, segs, hyp)
    if strip is None:
        det.reason = "degenerate neck strip"
        return det
    det.strip = strip
    prof = column_profile(strip)
    pos = prof[prof > 0]
    if len(pos) < 10:
        det.reason = "empty strip profile"
        return det
    thr = max(3.0, 0.25 * float(np.percentile(pos, 97)))
    xs, hs = find_peaks(prof, thr)
    det.debug["profile"], det.debug["peaks"] = prof, (xs, hs)
    valid = np.flatnonzero(prof != 0)
    grid = fit_grid_1d(xs, hs, (valid[0], valid[-1]), seed=P.seed)
    if grid is None or len(grid.peak_idx) < 5:
        det.reason = "no fret grid in the strip"
        return det
    det.grid = grid
    det.debug["valid"] = (int(valid[0]), int(valid[-1]))
    return det


def _wire_stage(gray, det: Detection, P: DetectParams) -> None:
    grid, strip = det.grid, det.strip
    v0, v1 = det.debug["valid"]
    lo, hi = grid.index_range
    rough: dict[int, np.ndarray] = {}
    for n in range(lo - 3, hi + 4):
        x = float(grid.x(n))
        if not v0 + 2 <= x <= v1 - 2:
            continue
        rough[n] = strip.point(np.array([x, x]), np.array([-0.5, 0.5]))
    model, wires, info = _wires_to_model(gray, rough, set(range(lo, hi + 1)), strip.down, P)
    if model is None:
        det.reason = info.get("reason", "fit failed")
        return
    # Pass 2: re-predict every wire from the fitted homography (exact rectification) and re-measure.
    for _ in range(2):
        rough2 = _predict_wires(model, gray.shape, wires)
        core = {n for n, w in wires.items() if w.vis == 2}
        model2, wires2, info2 = _wires_to_model(gray, rough2, core, strip.down, P)
        if model2 is None:
            break
        model, wires, info = model2, wires2, info2
    det.model, det.wires, det.edges = model, wires, info.get("edges")
    det.debug["board_edges"], det.debug["strings"] = info.get("board_edges"), info.get("strings")
    det.debug["strings_source"], det.debug["strings_rho"] = info.get("strings_source"), info.get("strings_rho")
    det.anchors = info["anchors"]
    det.ok = True


def _predict_wires(model: FretboardModel, shape, wires: dict[int, WireObs]) -> dict[int, np.ndarray]:
    h, w = shape[:2]
    lo, hi = min(wires), max(wires)
    out = {}
    for n in range(lo - 3, hi + 4):
        if n in wires and wires[n].vis == 2:
            out[n] = wires[n].seg
            continue
        b, t = model.project_fy(np.array([n, n]), np.array([0.0, 1.0]))
        if not all(0 <= p[0] < w and 0 <= p[1] < h for p in (b, t)):
            continue
        out[n] = np.array([b, t])
    return out


def _grow_short_wires(gray, wires: dict[int, WireObs], nrm_down, P: DetectParams, min_ratio: float = 1.2) -> None:
    """Width prior on the wires themselves: a fret wire spans the strings, at least ~0.7x the fret spacing at fret 1
    and more further up. A visible wire shorter than `min_ratio` x the local spacing (the neck hypothesis only covered
    a bright binding / one string) is grown to the end of its ridge and re-snapped."""
    vis = sorted(n for n, w in wires.items() if w.vis == 2)
    mids = {n: 0.5 * (wires[n].seg[0] + wires[n].seg[1]) for n in vis}
    for i, n in enumerate(vis):
        sp = [np.linalg.norm(mids[m] - mids[n]) / abs(m - n) for m in vis[max(i - 1, 0) : i + 2] if m != n]
        if not sp:
            continue
        seg = wires[n].seg
        L = float(np.linalg.norm(seg[1] - seg[0]))
        if L >= min_ratio * float(np.mean(sp)):
            continue
        best = None
        for ext in (1.0, 2.5, 6.0, 12.0):
            e = wire_extent(gray, seg[0], seg[1], ext=ext)
            if e is not None:
                best = np.array(e)
                break
        if best is None or np.linalg.norm(best[1] - best[0]) <= L:
            continue
        sn, _ = snap_wire_fast(gray, best[0], best[1], search=2.0, step=0.5)
        med, sup = wire_support(gray, sn[0], sn[1])
        if sup >= P.vis_support and med >= P.min_wire_score:
            w = wires[n]
            w.seg, w.score, w.support = _bass_first(sn, nrm_down), med, sup


def _outer_strings(gray, model, keep2: dict, board, index_shift: int | None, n_frets: int = 24):
    """Outer-string lines (low E, high E) consistent with the width prior, first that fits: the string comb, the board
    edges inset by EDGE_INSET, the prior span centred on the board. Returns (lines, source, StringFit | None, rho)."""
    frets = sorted(keep2)
    sf = detect_strings(gray, model, frets, shift=index_shift, n_frets=n_frets)
    if sf is not None and sf.ok:
        return sf.lines, "comb", sf, sf.rho
    segs = {n: keep2[n][0] for n in frets}
    shifts = allowed_shifts(frets, index_shift, n_frets)
    inset = inset_lines(*board, [segs[n] for n in frets])
    rho = lines_rho(model, inset, segs, shifts)
    ok = prior_ok(rho)
    if ok.any():
        k = int(np.argmin(np.abs(rho - RHO_NOM).sum(-1) + 10.0 * ~ok))
        return inset, "edge_inset", sf, (float(rho[k, 0]), float(rho[k, 1]))
    # Neither is plausible: the prior span at the shift whose board-edge width is closest to nominal, centred on the
    # board (flagged downstream; the wires/numbering are still right, only the cross-neck extent is a guess).
    # Centre: the median plane position of the wire midpoints (the wires span the strings; the board edges may not).
    rb = lines_rho(model, board, segs, shifts)
    s = int(shifts[int(np.argmin(np.abs(rb / 1.2 - RHO_NOM).sum(-1)))]) if index_shift is None else int(index_shift)
    mids = np.array([0.5 * (segs[n][0] + segs[n][1]) for n in frets])
    v_c = float(np.median(apply_h(np.linalg.inv(model.H), mids)[:, 1]))
    return prior_lines(model, frets, s, v_c=v_c), "prior", sf, (RHO_NOM, RHO_NOM)


def _wires_to_model(
    gray, rough: dict[int, np.ndarray], core: set[int], nrm_down, P: DetectParams, index_shift: int | None = None
):
    """Snap the rough wires, decide visibility, measure the board edges, fit the fretboard model."""
    wires: dict[int, WireObs] = {}
    for n, seg in sorted(rough.items()):
        if np.linalg.norm(seg[1] - seg[0]) < 8:  # degenerate (e.g. a wire predicted at the image border)
            continue
        sn, score = snap_wire_fast(gray, seg[0], seg[1], search=P.snap_search, step=0.5)
        sn = _bass_first(sn, nrm_down)
        med, sup = wire_support(gray, sn[0], sn[1])
        need = P.vis_support if n in core else P.extend_support
        if sup >= need and med >= P.min_wire_score:
            wires[n] = WireObs(n, sn, med, sup, 2)
        elif n in core:
            wires[n] = WireObs(n, _bass_first(seg, nrm_down), med, sup, 1)
    vis2 = {n: (w.seg, 2) for n, w in wires.items() if w.vis == 2}
    if len(vis2) < 4:
        return None, wires, {"reason": "fewer than 4 visible wires"}
    _grow_short_wires(gray, wires, nrm_down, P)
    vis2 = {n: (w.seg, 2) for n, w in wires.items() if w.vis == 2}
    try:
        anchors, info = anchors_from_segments(gray, vis2, nrm_down)
    except (FitError, np.linalg.LinAlgError) as e:
        return None, wires, {"reason": str(e)}
    try:
        model = fit_fretboard(anchors, tau=None)
    except FitError as e:
        return None, wires, {"reason": str(e)}
    # D13: anchors on the outer strings. The board edges above give a first model; find the string comb on it.
    keep2 = {n: v for n, v in vis2.items() if n not in info["rejected"]}
    board = info["lines"]
    lines, src, sf, rho = _outer_strings(gray, model, keep2, board, index_shift)
    try:
        anchors, info2 = anchors_from_segments(gray, keep2, nrm_down, edge_lines=lines)
        model = fit_fretboard(anchors, tau=None)
    except (FitError, np.linalg.LinAlgError) as e:
        return None, wires, {"reason": f"string refit: {e}"}
    info.update(lines=info2["lines"], board_edges=board, strings=sf, strings_source=src, strings_rho=rho)
    for n in info["rejected"]:
        wires.pop(n, None)
    bad = {a.fret for a in model.outliers}
    for n in bad:
        if n in wires:
            wires[n].vis = 1
    # Occluded wires: predicted from the fit, intersected with the measured edges.
    be, te = info["lines"]
    out = [a for a in anchors if a.fret not in bad]
    for n, w in sorted(wires.items()):
        if w.vis == 2:
            continue
        b, t = model.project_fy(np.array([n, n]), np.array([0.0, 1.0]))
        wl = (b, (t - b) / np.linalg.norm(t - b))
        pb, pt = intersect(wl, be), intersect(wl, te)
        w.seg = np.array([pb, pt])
        out += [A.Anchor(n, "bass", *map(float, pb), 1), A.Anchor(n, "treble", *map(float, pt), 1)]
    info["anchors"] = sorted(out, key=lambda a: (a.fret, a.side))
    return model, wires, info


def angle_deg(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def refit_subset(
    gray: np.ndarray,
    det: Detection,
    keep: set[int],
    params: DetectParams | None = None,
    index_shift: int | None = None,
) -> Detection:
    """Re-measure a detection using only the wires in `keep` (e.g. after numbering drops wires past the last fret or
    ones the AI flagged as the fingerboard end; with the numbering's `index_shift` the width prior is exact).
    Returns a new Detection (or the input if the refit fails)."""
    P = params or DetectParams()
    rough = {n: w.seg for n, w in det.wires.items() if n in keep}
    core = {n for n, w in det.wires.items() if n in keep}
    model, wires, info = _wires_to_model(gray, rough, core, det.strip.down, P, index_shift)
    if model is None:
        return det
    out = Detection(
        True,
        anchors=info["anchors"],
        model=model,
        wires=wires,
        edges=info.get("edges"),
        grid=det.grid,
        strip=det.strip,
        hyp=det.hyp,
        n_candidates=det.n_candidates,
        debug={
            **det.debug,
            "board_edges": info.get("board_edges"),
            "strings": info.get("strings"),
            "strings_source": info.get("strings_source"),
            "strings_rho": info.get("strings_rho"),
        },
    )
    return out
