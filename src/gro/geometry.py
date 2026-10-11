"""Fretboard geometry: fit a plane->image homography from anchors and derive every fret and string line.

Canonical fretboard coordinates (PLAN §1.1 B):
  x = u = distance from the nut / scale length. Fret n is at u_n = 1 - 2^(-n/12) (u_12 = 0.5).
  y = 0 at the bass edge .. 1 at the treble edge.
The fretboard tapers, so (u, y) is not itself a plane coordinate. The physical plane point is
  P(u, y) = (u, (y - 0.5) * (1 + tau * u))
in units where the nut width is 1 (the absolute nut width is absorbed into H). `H` maps P to image pixels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from gro.anchors import Anchor

DEFAULT_TAU = 0.30  # half-width taper: 43 mm nut -> ~56 mm at the 24th fret
NUT_WIDTH_U = 43.0 / 648.0  # nut width as a fraction of scale length (for drawing round markers)
# D13: anchors (y = 0 / 1) sit on the outer strings, so the strings span exactly y in [0, 1]. The visible board
# extends about half a string spacing beyond them (BOARD_PAD, used for the drawn cells).
STRING_MARGIN = 0.0
BOARD_PAD = 0.1
N_STRINGS = 6


def fret_u(n: float | np.ndarray) -> float | np.ndarray:
    """Normalized distance from the nut to fret `n` (12th-root-of-2 spacing)."""
    return 1.0 - np.power(2.0, -np.asarray(n, dtype=float) / 12.0)


def plane_point(u, y, tau: float = DEFAULT_TAU) -> np.ndarray:
    """Canonical (u, y) -> physical plane point(s), shape (..., 2)."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    u, y = np.broadcast_arrays(u, y)
    return np.stack([u, (y - 0.5) * (1.0 + tau * u)], axis=-1)


def string_y(s: int | np.ndarray, n_strings: int = N_STRINGS, margin: float = STRING_MARGIN):
    """Canonical y of string s (0 = low E at the bass edge side .. n-1 = high E)."""
    return margin + (1.0 - 2.0 * margin) * np.asarray(s, dtype=float) / (n_strings - 1)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    shp = pts.shape
    p = pts.reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return (q[:, :2] / q[:, 2:3]).reshape(shp)


def mirror_matrix(width: int) -> np.ndarray:
    """M = [[-1, 0, W-1], [0, 1, 0], [0, 0, 1]] (D1). M is its own inverse."""
    return np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------------------------------------------
# Homography estimation


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return T, apply_h(T, pts)


def dlt(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted, Hartley-normalized DLT homography src -> dst (N >= 4)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if w is None:
        w = np.ones(len(src))
    Ts, s = _normalize(src)
    Td, d = _normalize(dst)
    rows = []
    for (x, y), (u, v), wi in zip(s, d, np.sqrt(w), strict=True):
        rows.append(wi * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
        rows.append(wi * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _refine(H: np.ndarray, src: np.ndarray, dst: np.ndarray, w: np.ndarray, huber_px: float, iters: int = 15):
    """Gauss-Newton / IRLS on reprojection error with a Huber loss (8 params, H[2,2] = 1)."""
    h = (H / H[2, 2]).ravel()[:8].copy()

    def residuals(hv):
        Hm = np.append(hv, 1.0).reshape(3, 3)
        return (apply_h(Hm, src) - dst).ravel()

    for _ in range(iters):
        r = residuals(h)
        rn = np.sqrt(r[0::2] ** 2 + r[1::2] ** 2)
        hub = np.where(rn <= huber_px, 1.0, huber_px / np.maximum(rn, 1e-12))
        wr = np.repeat(w * hub, 2)
        J = np.empty((len(r), 8))
        for k in range(8):
            eps = 1e-7 * max(1.0, abs(h[k]))
            hp = h.copy()
            hp[k] += eps
            J[:, k] = (residuals(hp) - r) / eps
        A = J.T @ (J * wr[:, None])
        g = J.T @ (wr * r)
        try:
            step = np.linalg.solve(A + 1e-9 * np.eye(8) * np.trace(A), -g)
        except np.linalg.LinAlgError:
            break
        h = h + step
        if np.abs(step).max() < 1e-12:
            break
    return np.append(h, 1.0).reshape(3, 3)


@dataclass
class FretboardModel:
    """Fitted fretboard: canonical (u, y) -> image."""

    H: np.ndarray
    tau: float = DEFAULT_TAU
    n_frets: int = 24
    n_strings: int = N_STRINGS
    rmse_px: float = float("nan")
    inliers: list[Anchor] = field(default_factory=list)
    outliers: list[Anchor] = field(default_factory=list)
    index_shift: int = 0
    index_margin: float = float("inf")  # cost ratio 2nd-best / best shift (higher = more certain)

    def project(self, u, y) -> np.ndarray:
        return apply_h(self.H, plane_point(u, y, self.tau))

    def project_fy(self, fret, y) -> np.ndarray:
        """Project points given by (fractional) fret number and canonical y."""
        return self.project(fret_u(fret), y)

    def fret_line(self, n: float) -> np.ndarray:
        """Image endpoints [bass, treble] of fret wire n (0 = nut)."""
        return self.project_fy(np.array([n, n]), np.array([0.0, 1.0]))

    def string_line(self, s: int, fret_from: float = 0.0, fret_to: float | None = None) -> np.ndarray:
        fret_to = self.n_frets if fret_to is None else fret_to
        y = float(string_y(s, self.n_strings))
        return self.project_fy(np.array([fret_from, fret_to]), np.array([y, y]))

    def cell_quad(self, n: int, y0: float = 0.0, y1: float = 1.0, inset_u: float = 0.0) -> np.ndarray:
        """Image quad of the cell between wires n-1 and n, across-neck from y0 to y1; inset_u in fractions of the
        cell length (gap so the wires show through)."""
        ua, ub = fret_u(n - 1), fret_u(n)
        d = (ub - ua) * inset_u
        ua, ub = ua + d, ub - d
        return self.project(np.array([ua, ub, ub, ua]), np.array([y0, y0, y1, y1]))

    def fret_width_px(self, n: float, y: float = 0.5) -> float:
        """Local spacing in px between wire n and wire n+1 (used for normalized errors)."""
        p = self.project_fy(np.array([n, n + 1]), np.array([y, y]))
        return float(np.linalg.norm(p[1] - p[0]))

    def fret_widths_px(self, n: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized `fret_width_px`."""
        n, y = np.broadcast_arrays(np.asarray(n, float), np.asarray(y, float))
        return np.linalg.norm(self.project_fy(n + 1, y) - self.project_fy(n, y), axis=-1)

    def visible_frets(self, width: int, height: int, pad: float = 0.0) -> list[int]:
        """Fret wires 0..n_frets whose midpoint projects inside the image (with orientation sanity)."""
        out = []
        for n in range(self.n_frets + 1):
            m = self.project_fy(n, 0.5)
            if -pad <= m[0] < width + pad and -pad <= m[1] < height + pad:
                out.append(n)
        return out

    def residuals(self, anchors: Iterable[Anchor]) -> np.ndarray:
        anchors = list(anchors)
        if not anchors:
            return np.zeros(0)
        pred = self.project_fy(np.array([a.fret for a in anchors]), np.array([a.y_canon for a in anchors]))
        obs = np.array([[a.x, a.y] for a in anchors])
        return np.linalg.norm(pred - obs, axis=1)

    def normalized_residuals(self, anchors: Iterable[Anchor]) -> np.ndarray:
        anchors = list(anchors)
        r = self.residuals(anchors)
        fw = np.array([max(self.fret_width_px(a.fret, a.y_canon), 1e-6) for a in anchors])
        return r / fw


class FitError(ValueError):
    pass


def _correspondences(anchors: Sequence[Anchor], tau: float, shift: int = 0):
    src = plane_point(
        fret_u(np.array([a.fret + shift for a in anchors], float)),
        np.array([a.y_canon for a in anchors]),
        tau,
    )
    dst = np.array([[a.x, a.y] for a in anchors], float)
    w = np.array([a.conf * (1.0 if a.vis == 2 else 0.5) for a in anchors], float)
    return src, dst, w


def _degenerate(anchors: Sequence[Anchor]) -> bool:
    frets = {a.fret for a in anchors}
    sides = {a.side for a in anchors}
    return len(frets) < 2 or len(sides) < 2


def _sample_ok(sample: Sequence[Anchor]) -> bool:
    """Minimal sample: 2+ distinct frets, both edges, no duplicate slots, and no 3 points on one edge."""
    if _degenerate(sample) or len({(a.fret, a.side) for a in sample}) < len(sample):
        return False
    return all(sum(a.side == side for a in sample) <= 2 for side in ("bass", "treble"))


def fit_fretboard(
    anchors: Sequence[Anchor],
    tau: float | None = DEFAULT_TAU,
    n_frets: int = 24,
    ransac_iters: int = 300,
    inlier_frac: float = 0.15,
    min_inlier_px: float = 2.0,
    shift: int = 0,
    seed: int = 0,
) -> FretboardModel:
    """Robustly fit the fretboard model to anchors with known fret indices (+ optional `shift`).

    RANSAC over minimal 4-point samples (>= 2 fret lines, both edges), inlier threshold = `inlier_frac` x local fret
    width (>= `min_inlier_px`), then Huber-IRLS refinement on the inliers. If `tau` is None it is grid-searched (only
    meaningful with many frets spread along the neck).
    """
    anchors = [a for a in anchors if a.vis > 0]
    if len(anchors) < 4 or _degenerate(anchors):
        raise FitError("need >= 4 anchors on >= 2 fret lines covering both edges")
    if tau is None:
        best = None
        for t in np.linspace(0.0, 0.6, 13):
            m = fit_fretboard(anchors, float(t), n_frets, ransac_iters, inlier_frac, min_inlier_px, shift, seed)
            if best is None or (len(m.inliers), -m.rmse_px) > (len(best.inliers), -best.rmse_px):
                best = m
        return best

    src, dst, w = _correspondences(anchors, tau, shift)
    n = len(anchors)
    a_fret = np.array([a.fret + shift for a in anchors], float)
    a_y = np.array([a.y_canon for a in anchors])
    rng = np.random.default_rng(seed)

    def score(H):
        m = FretboardModel(H, tau, n_frets)
        try:
            r = np.linalg.norm(apply_h(H, src) - dst, axis=1)
            fw = m.fret_widths_px(a_fret, a_y)
        except FloatingPointError:
            return None
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(fw)):
            return None
        thr = np.maximum(inlier_frac * fw, min_inlier_px)
        inl = r < thr
        cost = np.sum(np.minimum(r / thr, 1.0) ** 2)  # MSAC
        return inl, cost

    best_inl, best_cost = None, np.inf
    if n == 4:
        best_inl = np.ones(n, bool)
    else:
        idx = np.arange(n)
        with np.errstate(all="ignore"):
            for _ in range(ransac_iters):
                pick = rng.choice(idx, 4, replace=False)
                if not _sample_ok([anchors[i] for i in pick]):
                    continue
                try:
                    H = dlt(src[pick], dst[pick])
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(H)):
                    continue
                sc = score(H)
                if sc is None:
                    continue
                inl, cost = sc
                if inl.sum() >= 4 and cost < best_cost:
                    best_inl, best_cost = inl, cost
        if best_inl is None or _degenerate([anchors[i] for i in np.flatnonzero(best_inl)]):
            best_inl = np.ones(n, bool)

    # Refine on inliers and re-evaluate the inlier set until it is stable.
    for _ in range(10):
        sel = np.flatnonzero(best_inl)
        H = dlt(src[sel], dst[sel], w[sel])
        m0 = FretboardModel(H, tau, n_frets)
        huber = max(min_inlier_px, 0.1 * np.median(m0.fret_widths_px(a_fret[sel], 0.5)))
        H = _refine(H, src[sel], dst[sel], w[sel], huber)
        sc = score(H)
        if sc is None:
            break
        new_inl = sc[0]
        if (
            new_inl.sum() < 4
            or _degenerate([anchors[i] for i in np.flatnonzero(new_inl)])
            or (new_inl == best_inl).all()
        ):
            break
        best_inl = new_inl

    sel = np.flatnonzero(best_inl)
    r = np.linalg.norm(apply_h(H, src[sel]) - dst[sel], axis=1)
    return FretboardModel(
        H=H,
        tau=tau,
        n_frets=n_frets,
        rmse_px=float(np.sqrt(np.mean(r**2))),
        inliers=[anchors[i] for i in sel],
        outliers=[anchors[i] for i in range(n) if not best_inl[i]],
        index_shift=shift,
    )


def metric_cost(H: np.ndarray, image_size: tuple[int, int], focal: float | None = None) -> float:
    """How far H is from the image of a *metric* plane under a pinhole camera (square pixels, centered principal
    point). 0 = perfectly consistent. With `focal=None` the best focal in [0.3, 3] x width is used.

    Uses the physical aspect of the fretboard (nut width / scale length = NUT_WIDTH_U): H . diag(1, 1/NUT_WIDTH_U, 1)
    maps metric plane coords (in scale lengths) to pixels, and K^-1 of that must have orthogonal, equal-norm first two
    columns (they are r1, r2 of the camera rotation, times a scale).
    """
    w, h = image_size
    G = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]]) @ H @ np.diag([1.0, 1.0 / NUT_WIDTH_U, 1.0])
    fs = [focal] if focal is not None else np.geomspace(0.3 * w, 3.0 * w, 120)
    best = np.inf
    for f in fs:
        B = G * np.array([[1 / f], [1 / f], [1.0]])
        b1, b2 = B[:, 0], B[:, 1]
        n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
        best = min(best, float((b1 @ b2 / (n1 * n2)) ** 2 + np.log(n1 / n2) ** 2))
    return best


def resolve_index_shift(
    anchors: Sequence[Anchor],
    shifts: Iterable[int] = range(-6, 7),
    tau: float = DEFAULT_TAU,
    n_frets: int = 24,
    image_size: tuple[int, int] | None = None,
    focal: float | None = None,
    metric_weight: float = 1.0,
    **kw,
) -> FretboardModel:
    """Fit with every candidate index shift `s` (true fret = labelled fret + s) and keep the cheapest.

    This is the D4 "no nut" path. NOTE: fret spacing alone can NOT resolve the index. Fret positions are a geometric
    progression from the bridge (u_{n+s} = 1 - 2^(-s/12) (1 - u_n)), i.e. a shift is an *affine* map of u, which any
    homography absorbs; cross-ratios of fret positions are identical for every shift. Only the taper (weak) and the
    *metric* shape of the board break the tie: the ratio of fret gap to board width changes ~6% per fret, which is
    observable through H when the camera is roughly a pinhole with known-ish focal (`metric_cost`). Pass `image_size`
    to enable that cue. It is ~90% right on synthetic pinhole data, but on the real M1 frames it was biased by 2-3 frets
    (nut width / scale length assumptions, lens distortion, binding vs. board edge), so inlay and landmark cues must
    decide (PLAN D4). With tau = 0 the reprojection cost is exactly shift-invariant.

    Cost = mean squared normalized reprojection residual (px / local fret width) + metric_weight * metric_cost.
    The returned model has `index_shift` set and `index_margin` = cost(2nd best) / cost(best).
    """
    anchors = [a for a in anchors if a.vis > 0]
    if len({a.fret for a in anchors}) < 4:
        raise FitError("index resolution needs >= 4 distinct fret wires")
    lo = min(a.fret for a in anchors)
    hi = max(a.fret for a in anchors)
    results = []
    for s in shifts:
        if lo + s < 0 or hi + s > n_frets:
            continue
        try:
            m = fit_fretboard(anchors, tau, n_frets, shift=s, **kw)
        except FitError:
            continue
        shifted = [Anchor(a.fret + s, a.side, a.x, a.y, a.vis, a.conf) for a in anchors]
        nr = m.normalized_residuals(shifted)
        cost = float(np.mean(np.minimum(nr, 0.5) ** 2)) + 1e-9
        if image_size is not None:
            cost += metric_weight * metric_cost(m.H, image_size, focal)
        results.append((cost, s, m))
    if not results:
        raise FitError("no feasible index shift")
    results.sort(key=lambda t: t[0])
    cost, s, m = results[0]
    # Re-express the model so callers can use true fret numbers directly.
    m.index_shift = s
    m.index_margin = results[1][0] / cost if len(results) > 1 else float("inf")
    m.inliers = [Anchor(a.fret + s, a.side, a.x, a.y, a.vis, a.conf) for a in m.inliers]
    m.outliers = [Anchor(a.fret + s, a.side, a.x, a.y, a.vis, a.conf) for a in m.outliers]
    return m


def model_from_h(H, tau: float = DEFAULT_TAU, n_frets: int = 24) -> FretboardModel:
    """Build a model from an explicit plane->image H (3x3)."""
    H = np.asarray(H, float)
    return FretboardModel(H / H[2, 2], tau, n_frets)


def mirror_model(model: FretboardModel, width: int) -> FretboardModel:
    """Model for the mirrored image: H_img = M . H_canon (D1)."""
    out = FretboardModel(**{**model.__dict__})
    out.H = mirror_matrix(width) @ model.H
    return out


def transform_model(model: FretboardModel, G: np.ndarray) -> FretboardModel:
    """Compose an image-space homography (e.g. frame-to-frame KLT motion) onto the model."""
    out = FretboardModel(**{**model.__dict__})
    H = G @ model.H
    out.H = H / H[2, 2]
    return out
