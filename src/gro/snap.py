"""Annotation assist (PLAN §3.2 "snap"): refine rough fret-wire clicks onto the bright wire ridge, and build anchors
as intersections of wire lines with the fretboard edge lines.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from gro.anchors import Anchor


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = cv2.remap(
        gray,
        pts[:, 0].astype(np.float32).reshape(1, -1),
        pts[:, 1].astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return m.ravel().astype(float)


def ridge_score(gray: np.ndarray, p0, p1, t0: float = 0.15, t1: float = 0.85, n: int = 40, side_px: float = 3.0):
    """Mean ridge response along the segment: centre brightness minus the mean of both flanks. Uses the median of
    per-sample responses so a finger or inlay crossing part of the wire doesn't dominate."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    t = np.linspace(t0, t1, n)[:, None]
    c = p0 + t * d
    resp = _sample(gray, c) - 0.5 * (_sample(gray, c + side_px * nrm) + _sample(gray, c - side_px * nrm))
    return float(np.median(resp))


def snap_wire(gray: np.ndarray, p0, p1, search: float = 4.0, step: float = 0.5, **kw) -> tuple[np.ndarray, float]:
    """Move each endpoint of a rough wire independently along the across-wire direction (±search px) to maximise the
    ridge score. Returns the refined endpoints (2x2) and the score."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    offs = np.arange(-search, search + 1e-9, step)
    best = (-np.inf, p0, p1)
    for a in offs:
        for b in offs:
            q0, q1 = p0 + a * nrm, p1 + b * nrm
            s = ridge_score(gray, q0, q1, **kw)
            if s > best[0]:
                best = (s, q0, q1)
    return np.array([best[1], best[2]]), best[0]


def wire_extent(
    gray: np.ndarray, p0, p1, ext: float = 0.35, step: float = 0.25, side_px: float = 3.0, frac: float = 0.45
) -> tuple[np.ndarray, np.ndarray] | None:
    """Endpoints of the bright wire ridge along the line p0->p1 (extended by `ext` on both sides): where the ridge
    response, walking outwards from the middle, first drops below `frac` x its median over the middle 50%."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]]) / max(L, 1e-9)
    t = np.arange(-ext, 1 + ext + 1e-9, step / L)
    c = p0 + t[:, None] * d
    r = _sample(gray, c) - 0.5 * (_sample(gray, c + side_px * nrm) + _sample(gray, c - side_px * nrm))
    r = np.convolve(r, np.ones(5) / 5, mode="same")
    mid = (t > 0.25) & (t < 0.75)
    level = np.median(r[mid])
    if level <= 2.0:
        return None
    thr = frac * level
    i_mid = int(np.argmin(np.abs(t - 0.5)))
    lo = i_mid
    while lo > 0 and r[lo - 1] > thr:
        lo -= 1
    hi = i_mid
    while hi < len(t) - 1 and r[hi + 1] > thr:
        hi += 1
    if lo == 0 or hi == len(t) - 1:
        return None  # ran off the search window: no clear end
    return c[lo], c[hi]


def robust_line(points, iters: int = 5, k: float = 2.5, min_tol: float = 1.0):
    """TLS line with iterative trimming of points farther than max(k * MAD-sigma, min_tol)."""
    p = np.asarray(points, float)
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        c, u = fit_line(p[keep])
        n = np.array([-u[1], u[0]])
        dist = np.abs((p - c) @ n)
        sigma = 1.4826 * np.median(dist[keep]) + 1e-9
        new = dist < max(k * sigma, min_tol)
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return fit_line(p[keep]), keep


def fit_line(points: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through points: (point_on_line, unit_direction)."""
    p = np.asarray(points, float)
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[0]


def intersect(l1, l2) -> np.ndarray:
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def anchors_from_wires(
    gray: np.ndarray,
    bass_edge: Sequence[Sequence[float]],
    treble_edge: Sequence[Sequence[float]],
    wires: dict[int, tuple[Sequence[float], Sequence[float], int]],
    snap: bool = True,
    no_snap: Sequence[int] = (0,),
    auto_edges: bool = True,
    **kw,
) -> tuple[list[Anchor], dict[int, dict]]:
    """wires: fret -> (rough point near the bass end, rough point near the treble end, vis). Returns anchors at the
    intersections of each (snapped) wire with the edge lines, plus per-wire debug info.

    Edge lines: if `auto_edges`, they are robustly fitted through the measured ends of the wire ridges (`wire_extent`),
    with the clicked edge points added as extra support; otherwise only the clicked points are used.
    """
    segs, info = {}, {}
    ends_b, ends_t = list(map(tuple, bass_edge)), list(map(tuple, treble_edge))
    for fret, (pb, pt, vis) in sorted(wires.items()):
        seg = np.array([pb, pt], float)
        score, extent = None, None
        if snap and vis == 2 and fret not in no_snap:  # the nut is a block, not a ridge
            seg, score = snap_wire(gray, seg[0], seg[1], **kw)
            if auto_edges:
                extent = wire_extent(gray, seg[0], seg[1])
                if extent is not None:
                    ends_b.append(tuple(extent[0]))
                    ends_t.append(tuple(extent[1]))
        segs[fret] = (seg, vis)
        info[fret] = {"rough": np.array([pb, pt], float), "snapped": seg, "score": score, "extent": extent}
    (be, kb), (te, kt) = robust_line(ends_b), robust_line(ends_t)
    info["edges"] = {"bass": (be, np.asarray(ends_b), kb), "treble": (te, np.asarray(ends_t), kt)}
    out = []
    for fret, (seg, vis) in segs.items():
        wl = (seg[0], (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0]))
        b, t = intersect(wl, be), intersect(wl, te)
        out.append(Anchor(fret, "bass", float(b[0]), float(b[1]), vis))
        out.append(Anchor(fret, "treble", float(t[0]), float(t[1]), vis))
    return out, info
