"""Overlay jitter from prediction caches (scripts/cache_preds.py), per tracker variant.

  python scripts/jitter_eval.py --preds a.npz [b.npz ...] [--frets 24 22 22] [--out eval/jitter.json]

Wire positions: every wire 1..last projected at mid-board (y = 0.5) in source pixels, counted where inside the frame.
- step: median |p_t - p_{t-1}| on static-ish frames (smoothed wire velocity < 2% of the local fret width per frame;
  smoothing = centered moving average of the variant's own track over ~0.2 s).
- hf: median |p_t - MA(p)_t| over all frames (high-frequency residual after a centered ~0.2 s moving average).
- lag: median |p_t - MA(raw)_t| on fast frames (velocity of the moving-averaged raw fit > 5% of the fret width per
  frame): distance from the centered (non-causal) average of the unsmoothed fit, i.e. how far a causal smoother
  trails during fast moves (the raw fit's value is its own noise).
Each is reported in px and in % of the local fret width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_model import EmaTracker, SmoothTracker, anchors_from  # noqa: E402


def track(kpts, variant, fps, shape, n_frets, **kw):
    tr = {
        "raw": lambda: SmoothTracker(fps, smooth=False),
        "v1_ema": lambda: EmaTracker(),
        "one_euro": lambda: SmoothTracker(fps, **kw),
    }[variant]()
    out = []
    for k in kpts:
        if variant == "v1_ema":
            out.append(tr.update(anchors_from(k)))
        else:
            out.append(tr.update(anchors_from(k), shape, n_frets))
    return out


def wires(models, n_frets):
    """(T, n_frets) x and y of wire midpoints, and local fret width; NaN where no model."""
    T = len(models)
    P = np.full((T, n_frets, 2), np.nan)
    W = np.full((T, n_frets), np.nan)
    f = np.arange(1, n_frets + 1)
    for t, m in enumerate(models):
        if m is None:
            continue
        P[t] = m.project_fy(f.astype(float), np.full(n_frets, 0.5))
        W[t] = m.fret_widths_px(np.minimum(f, n_frets - 1) - 0.5, np.full(n_frets, 0.5))
    return P, W


def moving_average(P, win):
    k = win // 2
    out = np.full_like(P, np.nan)
    for t in range(k, len(P) - k):
        out[t] = P[t - k : t + k + 1].mean(0)  # NaN if any frame in the window lacks a model
    return out


def metrics(models, raw_models, n_frets, fps, shape):
    P, W = wires(models, n_frets)
    R, _ = wires(raw_models, n_frets)
    h, w = shape
    vis = (P[..., 0] >= 0) & (P[..., 0] < w) & (P[..., 1] >= 0) & (P[..., 1] < h)
    win = max(3, int(round(0.2 * fps)) | 1)
    MA = moving_average(P, win)
    step = np.linalg.norm(P[1:] - P[:-1], axis=-1)
    vel = np.linalg.norm(MA[1:] - MA[:-1], axis=-1)
    static = vis[1:] & vis[:-1] & (vel < 0.02 * W[1:])
    hf = np.linalg.norm(P - MA, axis=-1)
    hf_ok = vis & np.isfinite(hf)
    RMA = moving_average(R, win)
    rvel = np.linalg.norm(RMA[1:] - RMA[:-1], axis=-1)
    fast = vis[1:] & (rvel > 0.05 * W[1:])
    lag = np.linalg.norm(P[1:] - RMA[1:], axis=-1)

    def med(x, m, norm):
        if not m.any():
            return {"px": None, "pct_fret": None, "n": 0}
        return {
            "px": round(float(np.median(x[m])), 3),
            "pct_fret": round(float(np.median(x[m] / norm[m]) * 100), 2),
            "n": int(m.sum()),
        }

    return {
        "step_static": med(step, static, W[1:]),
        "hf": med(hf, hf_ok, W),
        "lag_fast": med(lag, fast, W[1:]),
        "has_fit": round(float(np.mean([m is not None for m in models])), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--frets", type=int, nargs="*", default=None)
    ap.add_argument("--min-cutoff", type=float, nargs="*", default=[1.0])
    ap.add_argument("--beta", type=float, nargs="*", default=[0.5])
    ap.add_argument("--floor", type=float, nargs="*", default=[0.0])
    ap.add_argument("--variants", nargs="*", default=["raw", "v1_ema", "one_euro"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    report = {}
    for i, p in enumerate(a.preds):
        d = np.load(p)
        fps, shape = float(d["fps"]), tuple(int(v) for v in d["size"][::-1])
        n = a.frets[i] if a.frets else 24
        kpts = d["kpts"]
        raw = track(kpts, "raw", fps, shape, n)
        rows = {}
        for v in a.variants:
            grid = [] if v != "one_euro" else [(mc, b, f) for mc in a.min_cutoff for b in a.beta for f in a.floor]
            for mc, b, fl in grid if v == "one_euro" else [(None, None, None)]:
                kw = {} if mc is None else {"min_cutoff": mc, "beta": b, "speed_floor": fl}
                models = raw if v == "raw" else track(kpts, v, fps, shape, n, **kw)
                # v1 drew all 24 frets; measure every variant on the same wires 1..n
                name = v if mc is None else f"{v}(mc={mc},beta={b},floor={fl})"
                rows[name] = metrics(models, raw, n, fps, shape)
                s, hf, lag = rows[name]["step_static"], rows[name]["hf"], rows[name]["lag_fast"]
                print(
                    f"{p.stem:14s} {name:28s} step {s['px']}px/{s['pct_fret']}%  hf {hf['px']}px/{hf['pct_fret']}%  "
                    f"lag {lag['px']}px/{lag['pct_fret']}% (n={lag['n']})  fit {rows[name]['has_fit']}",
                    flush=True,
                )
        report[p.stem] = {"fps": fps, "size": list(shape[::-1]), "frets": n, "variants": rows}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
