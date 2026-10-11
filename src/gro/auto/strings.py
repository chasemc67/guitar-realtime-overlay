"""String detection (PLAN D13): anchors sit where each fret wire crosses the outermost strings (low-E on the bass side,
high-E on the treble side), not on the board edges.

Given a first fretboard model (any across-neck convention: the board edges, or a previous string fit), the board is
resampled in its *physical plane* coordinates (u along the neck, v across it; `gro.geometry.plane_point`), where every
string is a straight line. Each fret cell (between two visible wires) gives a median across-neck ridge profile p_k(v)
over its inner columns (fingers and inlays only cover part of a cell), soft-saturated so inlays don't dominate. Summing
the profiles along lines v = c + d (u - u_ref) gives a line-score table S(c, d); a wire-presence term penalises lines
where the wires have ended (a string must cross the wires, with wire on both sides of it). The six strings are a comb
v_i(u) = A(u) + i B(u), i = 0..5 (low E first), with A and B linear in u (string spread tapers towards the nut),
searched exhaustively on S; the outer teeth are refined to sub-sample precision and are the low-E and high-E lines.
If the comb evidence is weak, the caller falls back to an inset of the board edges (`inset_lines`).

Width prior (cross-neck fix, 2026-10-11). The outer-string span is tied to the along-neck scale: with a 648 mm scale
the low-E..high-E span grows from ~35.5 mm at the nut to ~52 mm at the 24th fret, so
rho(u) = span_px(u) / (px per scale length along the neck at u) / span_mm(u) * 648 is ~1 for a straight-on view
(0.80-1.15 on the GT and the GOOD batch-002 frames; the "sliver" BAD frames were 0.08-0.61). The comb search only
considers spans with rho in [RHO_LO, RHO_HI] at both ends of the visible run and a taper consistent with the physical
one (no drift). Before numbering the fret indices are relative (an index shift is an affine map of u), so the prior
is applied for every shift that keeps the visible wires on a 24-fret board.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gro.geometry import FretboardModel, apply_h, fret_u

N_STRINGS = 6
# Fallback when strings aren't found: outer strings inset from the board edges by this fraction of the board width
# (median over the GT frames, low-E vs the wire-end edge).
EDGE_INSET = 0.085
# Width prior (see the module docstring).
SCALE_MM = 648.0
SPAN_NUT_MM, SPAN_24_MM = 35.5, 52.0
RHO_LO, RHO_HI, RHO_NOM = 0.7, 1.35, 0.95
TAPER_TOL = 0.25  # rho(body end) / rho(nut end) of the visible run within 1 +- this (GOOD frames: 0.87-1.0)


@dataclass
class StringFit:
    lines: tuple  # ((point, unit_dir) low E, (point, unit_dir) high E) in image px
    comb: np.ndarray  # (a0, a1, b0, b1): v_i(u) = a0 + a1 (u - u_ref) + i (b0 + b1 (u - u_ref)), plane units
    u_ref: float
    score: float  # mean clipped, normalised ridge response per tooth per cell (0..1)
    agree: float  # fraction of cells with a profile peak within ~1 px of both outer teeth (diagnostic only)
    n_cells: int
    ok: bool
    image_lines: dict | None = None  # all six string lines in image px (debug / review)
    rho: tuple | None = None  # width prior ratio at the first / last visible wire (best index shift)


def _plane_to_img(model: FretboardModel, u, v) -> np.ndarray:
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    return apply_h(model.H, np.stack([u, v], -1))


def span_mm(n_abs) -> np.ndarray:
    """Physical low-E..high-E span (mm) at absolute fret position `n_abs` (fractional)."""
    return SPAN_NUT_MM + (SPAN_24_MM - SPAN_NUT_MM) * fret_u(np.asarray(n_abs, float)) / fret_u(24)


def allowed_shifts(frets, shift: int | None = None, n_frets: int = 24) -> np.ndarray:
    """Index shifts (absolute = relative + shift) the prior is evaluated for: the known one, or every shift that keeps
    the visible wires within 0..n_frets + 2 (detections can include a board-end edge past the last fret)."""
    if shift is not None:
        return np.array([int(shift)])
    lo, hi = min(frets), max(frets)
    s = np.arange(-lo, n_frets + 2 - hi + 1)
    return s if len(s) else np.array([1 - lo])


def _local_scales(model: FretboardModel, n_rel: float, v_c: float = 0.0, eps: float = 1e-3):
    """(px per u along the neck, px per v across it) at relative fret position n_rel, across-neck position v_c."""
    u = float(fret_u(n_rel))
    p = _plane_to_img(model, [u - eps, u + eps, u, u], [v_c, v_c, v_c - eps, v_c + eps])
    return np.linalg.norm(p[1] - p[0]) / (2 * eps), np.linalg.norm(p[3] - p[2]) / (2 * eps)


def rho_factor(n_rel, shifts) -> np.ndarray:
    """rho = (span_px / px-per-relative-u) * rho_factor(n_rel, shift): converts relative u to the absolute fret
    position (d u_abs / d u_rel = 2^(-s/12)) and divides by the physical span in scale lengths."""
    n_rel = np.asarray(n_rel, float)
    s = np.asarray(shifts, float)
    return SCALE_MM / (np.power(2.0, s / 12.0) * span_mm(n_rel + s))


def lines_rho(model: FretboardModel, lines, segs_by_fret: dict, shifts) -> np.ndarray:
    """rho of two image lines (low E, high E) measured along the first and last wire: shape (len(shifts), 2)."""
    from gro.snap import intersect

    out = []
    for n in (min(segs_by_fret), max(segs_by_fret)):
        sg = segs_by_fret[n]
        wl = (sg[0], (sg[1] - sg[0]) / max(np.linalg.norm(sg[1] - sg[0]), 1e-9))
        try:
            w = float(np.linalg.norm(intersect(wl, lines[1]) - intersect(wl, lines[0])))
        except np.linalg.LinAlgError:
            w = 0.0
        su, _ = _local_scales(model, n)
        out.append(w / max(su, 1e-9) * rho_factor(n, shifts))
    return np.stack(out, -1)


def prior_ok(rho: np.ndarray, lo: float = RHO_LO, hi: float = RHO_HI, taper_tol: float = TAPER_TOL) -> np.ndarray:
    """rho (..., 2) at the first / last wire -> within the prior and taper-consistent."""
    r0, r1 = rho[..., 0], rho[..., 1]
    ratio = r1 / np.maximum(r0, 1e-9)
    return (r0 >= lo) & (r0 <= hi) & (r1 >= lo) & (r1 <= hi) & (np.abs(ratio - 1) <= taper_tol)


def prior_lines(model: FretboardModel, frets: list[int], shift: int, rho: float = RHO_NOM, v_c: float = 0.0):
    """Last-resort outer strings: centred on the model's centre line, span from the prior (rho) at `shift`."""
    lo, hi = min(frets), max(frets)
    vs = []
    for n in (lo, hi):
        su, sv = _local_scales(model, n, v_c)
        vs.append(0.5 * rho / float(rho_factor(n, shift)) * su / max(sv, 1e-9))
    us = fret_u(np.array([lo, hi], float))
    out = []
    for sgn in (-1, 1):
        p = _plane_to_img(model, us, v_c + sgn * np.array(vs))
        d = (p[1] - p[0]) / max(np.linalg.norm(p[1] - p[0]), 1e-9)
        out.append((p[0], d))
    return tuple(out)


def _cell_profiles(gray, model: FretboardModel, frets: list[int], v: np.ndarray, flank_px: float):
    """Per cell (n, n+1 both visible): median over inner columns of the across-neck ridge response at v."""
    from gro.auto.detect import sample

    out = []
    fs = set(frets)
    for n in sorted(fs):
        if n + 1 not in fs:
            continue
        u0, u1 = fret_u(n), fret_u(n + 1)
        us = u0 + (u1 - u0) * np.linspace(0.22, 0.78, 9)
        U, V = np.meshgrid(us, v, indexing="ij")
        pts = _plane_to_img(model, U, V)  # 9, nv, 2
        # Local image px per v unit (at the cell centre) -> flank offset in samples.
        c = pts[4]
        step_px = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        if step_px <= 1e-6:
            continue
        k = max(1, int(round(flank_px / step_px)))
        g = sample(gray, pts)
        r = np.full_like(g, np.nan)
        r[:, k:-k] = g[:, k:-k] - 0.5 * (g[:, : -2 * k] + g[:, 2 * k :])
        with np.errstate(all="ignore"):
            p = np.nan_to_num(np.nanmedian(np.where(np.isnan(r).all(axis=0), 0.0, r), axis=0), nan=0.0)
        # Normalise by the 6th-highest local maximum (~ the strings; inlays would dominate a percentile).
        pk = p[1:-1][(p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])]
        scale = max(float(np.sort(pk)[-N_STRINGS]) if len(pk) >= N_STRINGS else 0.0, 2.0)
        # Soft saturation (monotonic, so a strong string keeps a sharp maximum instead of a clipped plateau).
        out.append((0.5 * (u0 + u1), np.maximum(1.5 * np.tanh(p / scale / 1.5), -0.5), step_px))
    return out


def _wire_presence(gray, model, frets, v, cs_, ds_, u_ref, flank_px: float = 2.5, side_px: float = 2.5) -> np.ndarray:
    from gro.auto.detect import sample

    prof = []
    for n in sorted(set(frets)):
        u = fret_u(n)
        pts = _plane_to_img(model, np.full_like(v, u), v)
        nxt = _plane_to_img(model, np.full_like(v, fret_u(n + 0.5)), v)
        a = nxt - pts
        a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
        r = sample(gray, pts) - 0.5 * (sample(gray, pts + flank_px * a) + sample(gray, pts - flank_px * a))
        step_px = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        k = max(1, int(round(side_px / max(step_px, 1e-6))))
        mid = np.abs(v) < 0.25 * (1 + model.tau * u)
        lvl = max(float(np.median(r[mid])), 3.0)
        pres = np.clip(r / lvl / 0.4, 0.0, 1.0)  # >= 40% of the wire level counts as present
        # A string lies over the wire: the wire must continue on *both* sides of it (rejects the board edge /
        # binding highlight, where the wire ends).
        pp = np.pad(pres, k, mode="edge")
        pres = np.minimum(pp[: -2 * k], pp[2 * k :])
        prof.append((u - u_ref, pres))
    W = np.zeros((len(cs_), len(ds_)))
    for j, d in enumerate(ds_):
        for du, pres in prof:
            W[:, j] += np.interp(cs_ + d * du, v, pres, left=0.0, right=0.0)
    return W / max(len(prof), 1)


def _comb_score(prof, v, A, B):
    """Sum over teeth of the profile at A + i B (A, B arrays broadcast)."""
    i = np.arange(N_STRINGS).reshape((-1,) + (1,) * np.ndim(A))
    return np.interp(A + i * B, v, prof, left=-1.0, right=-1.0).sum(axis=0)


def detect_strings(
    gray: np.ndarray,
    model: FretboardModel,
    frets: list[int],
    flank_px: float = 2.0,
    shift: int | None = None,
    n_frets: int = 24,
) -> StringFit | None:
    """Find the six-string comb on the board of `model` (wires `frets` visible, relative indices; absolute = relative
    + `shift` if known). The comb span is restricted to the width prior. Returns None without >= 2 cells."""
    if len(frets) < 3:
        return None
    tau = model.tau
    u_all = fret_u(np.array(sorted(frets), float))
    half = 0.5 * (1 + tau * u_all.max())
    shifts = allowed_shifts(frets, shift, n_frets)
    n_lo, n_hi = float(min(frets)), float(max(frets))
    # Prior span (plane v units) at the first / last wire for every shift and rho in [RHO_LO, RHO_HI]; g = px per v
    # over px per u, so rho = span_v * g * rho_factor.
    g = np.array([(lambda a: a[1] / max(a[0], 1e-9))(_local_scales(model, n)) for n in (n_lo, n_hi)])
    fac = np.stack([rho_factor(n, shifts) for n in (n_lo, n_hi)], -1)  # S, 2
    span_lo = float(np.min(RHO_LO / (g * fac)))
    span_hi = float(np.max(RHO_HI / (g * fac)))
    # Search range: the first model's board (+-1.3 widths) and the whole prior span on either side of its centre.
    v_ext = max(1.3 * 2 * half, 1.15 * span_hi)
    # Sampling step ~0.2 px at the board centre (coarser for very wide searches).
    mid_u = float(np.median(u_all))
    w_px = float(np.linalg.norm(np.subtract(*_plane_to_img(model, [mid_u, mid_u], [-v_ext / 2, v_ext / 2]))))
    nv = int(np.clip(2 * w_px / 0.2, 300, 4000))
    v = np.linspace(-v_ext, v_ext, nv)
    cells = _cell_profiles(gray, model, frets, v, flank_px)
    if len(cells) < 2:
        return None
    u_c = np.array([c[0] for c in cells])
    u_ref = float(np.mean(u_c))
    du = u_c - u_ref
    P = np.array([c[1] for c in cells])  # K, nv
    # Line-score table over lines v = c + d (u - u_ref) in the plane: S(c, d) = sum over cells of the profile.
    dv = v[1] - v[0]
    # Line slope dv/du; outer strings diverge towards the body (plus model taper / parallax). In plane units of a
    # too-narrow first model the physical taper is a much larger slope, so the range scales with the prior span.
    d_max = 0.7 * max(1.0, span_hi / (2 * half))
    ds_ = np.linspace(-d_max, d_max, 57)
    cs_ = v.copy()
    S = np.zeros((len(cs_), len(ds_)))
    for j, d in enumerate(ds_):
        for k in range(len(cells)):
            S[:, j] += np.interp(cs_ + d * du[k], v, P[k], left=-0.5, right=-0.5)
    # Wire presence: strings lie over the wires, so a tooth where the wires have ended (off the board) is penalised.
    # Per wire, the along-neck ridge response vs v, max-filtered over +-2 px (the string itself breaks the ridge).
    W = _wire_presence(gray, model, frets, v, cs_, ds_, u_ref, flank_px=2.5)
    S = S - len(cells) * (1.0 - W)
    # Comb: tooth i at (c0 + i cs, d0 + i dd); even spacing across the neck at every u.
    c0 = np.arange(len(cs_))
    # Tooth spacing (index units) covering the prior spans at the reference u (the taper is checked below).
    sp = np.arange(max(int(0.8 * span_lo / 5 / dv), 2), int(1.2 * span_hi / 5 / dv) + 2)
    dstep = ds_[1] - ds_[0]
    u_ends = fret_u(np.array([n_lo, n_hi]))
    best = (-np.inf, None)
    dj = np.arange(len(ds_))
    for ddj in range(-6, 7):  # tooth slope step (index units)
        j0 = dj[(dj + 5 * ddj >= 0) & (dj + 5 * ddj < len(ds_)) & (dj >= 0)]
        if len(j0) == 0:
            continue
        # Width prior: comb span at the first / last wire for every spacing, any allowed shift.
        span = 5 * (sp[:, None] * dv + ddj * dstep * (u_ends[None, :] - u_ref))  # SP, 2
        rho = span[:, None, :] * g[None, None, :] * fac[None, :, :]  # SP, S, 2
        sp_ok = prior_ok(rho).any(axis=1)
        if not sp_ok.any():
            continue
        C0, SP, J0 = np.meshgrid(c0, sp[sp_ok], j0, indexing="ij")
        tot = np.zeros(C0.shape)
        valid = np.ones(C0.shape, bool)
        for i in range(N_STRINGS):
            ci = C0 + i * SP
            valid &= ci < len(cs_)
            tot += S[np.minimum(ci, len(cs_) - 1), J0 + i * ddj]
        tot[~valid] = -np.inf
        m = np.unravel_index(int(np.argmax(tot)), tot.shape)
        if tot[m] > best[0]:
            best = (float(tot[m]), (C0[m], SP[m], J0[m], ddj))
    if best[1] is None:
        return None
    ci0, spi, j0, ddj = best[1]
    teeth = []
    for i in range(N_STRINGS):
        teeth.append((cs_[ci0 + i * spi], ds_[j0 + i * ddj]))

    # Refine each outer tooth on its own (sub-sample c by a parabola, d by the table) near the comb position.
    def refine(c, d):
        jj = int(np.argmin(np.abs(ds_ - d)))
        cands = []
        for j in range(max(jj - 2, 0), min(jj + 3, len(ds_))):
            i0 = int(round((c - v[0]) / dv))
            r = max(int(0.2 * spi), 1)
            lo, hi = max(i0 - r, 1), min(i0 + r, len(cs_) - 2)
            i = lo + int(np.argmax(S[lo : hi + 1, j]))
            a, b, e = S[i - 1, j], S[i, j], S[i + 1, j]
            den = a - 2 * b + e
            off = 0.5 * (a - e) / den if abs(den) > 1e-9 else 0.0
            cands.append((b, cs_[i] + float(np.clip(off, -0.5, 0.5)) * dv, ds_[j]))
        _, c2, d2 = max(cands)
        return c2, d2

    teeth[0] = refine(*teeth[0])
    teeth[-1] = refine(*teeth[-1])
    score = best[0] / (len(cells) * N_STRINGS)
    # Agreement: fraction of cells whose profile peaks within ~1 px of both outer teeth.
    agree = []
    for k, (_, p, step_px) in enumerate(cells):
        ok_k = True
        for c, d in (teeth[0], teeth[-1]):
            x = c + d * du[k]
            r = max(1, int(round(1.0 / step_px)))
            i = int(round((x - v[0]) / dv))
            win = p[max(i - 3 * r, 0) : i + 3 * r + 1]
            ok_k &= len(win) > 0 and abs(int(np.argmax(win)) - min(3 * r, i)) <= r and win.max() > 0.3
        agree.append(ok_k)
    agree = float(np.mean(agree))
    comb = np.array([teeth[0][0], teeth[0][1], (teeth[-1][0] - teeth[0][0]) / 5, (teeth[-1][1] - teeth[0][1]) / 5])
    u_lo, u_hi = fret_u(min(frets)), fret_u(max(frets))
    alls = {}
    for i in range(N_STRINGS):
        us = np.array([u_lo, u_hi])
        vs = comb[0] + comb[1] * (us - u_ref) + i * (comb[2] + comb[3] * (us - u_ref))
        p = _plane_to_img(model, us, vs)
        d = (p[1] - p[0]) / max(np.linalg.norm(p[1] - p[0]), 1e-9)
        alls[i] = (p[0], d)
    lines = (alls[0], alls[N_STRINGS - 1])
    # Width prior on the refined outer teeth (refinement can move them a little).
    sp_ends = 5 * (comb[2] + comb[3] * (u_ends - u_ref))
    rho = sp_ends[None, :] * g[None, :] * fac  # S, 2
    in_prior = prior_ok(rho)
    k = int(np.argmin(np.abs(rho - RHO_NOM).sum(-1) + 10.0 * ~in_prior))
    ok = len(cells) >= 2 and score > 0.3 and bool(in_prior.any())
    fit = StringFit(lines, comb, u_ref, float(score), agree, len(cells), bool(ok), alls)
    fit.rho = (float(rho[k, 0]), float(rho[k, 1]))
    return fit


def inset_lines(be, te, segs: list[np.ndarray], frac: float = EDGE_INSET):
    """Fallback string lines: the board edges moved inwards by `frac` of the local board width (measured on the first
    and last wire)."""
    from gro.snap import intersect

    out = []
    for edge, other in ((be, te), (te, be)):
        pts = []
        for sg in (segs[0], segs[-1]):
            m = sg[0]
            d = (sg[1] - sg[0]) / np.linalg.norm(sg[1] - sg[0])
            a, b = intersect((m, d), edge), intersect((m, d), other)
            pts.append(a + frac * (b - a))
        d = (pts[1] - pts[0]) / max(np.linalg.norm(pts[1] - pts[0]), 1e-9)
        out.append((pts[0], d))
    return tuple(out)
