"""Apple-silicon speed: CoreML model latency per compute unit, and end-to-end FretboardTracker.process fps.

    python scripts/bench_coreml.py --models v1_fp16.mlpackage v1_int8.mlpackage --pt best.pt \
        --e2e-model v1_fp16.mlpackage --video 720p:videos/a.mp4:mirrored --video 1080p:videos/b.mp4 --out bench.json

Model latency: batch 1, constant random 640x640 input through the backend call (PIL conversion + predict), 20
warmups + --iters timed calls. End to end: --frames frames per clip, "decode excluded" = frames pre-decoded into RAM,
"decode included" = cv2.VideoCapture.read in the loop. Board-end detection on (no fret override), One-Euro on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gro.infer import FretboardTracker, load_backend  # noqa: E402
from gro.infer.backends import COMPUTE_UNITS  # noqa: E402


def stats(ms):
    ms = np.asarray(ms)
    return {"median_ms": round(float(np.median(ms)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2)}


def e2e(model, path, mirrored, n, units, start_s=10.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    out = {"shape": list(frames[0].shape[:2]), "frames": len(frames)}
    tracker = FretboardTracker(model, mirrored=mirrored, fps=fps, compute_units=units)
    for f in frames[:10]:
        tracker.process(f)
    tracker.reset()
    totals, parts = [], {}
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        s = tracker.process(f, i / fps)
        totals.append(s.timings_ms["total"])
        for k, v in s.timings_ms.items():
            parts.setdefault(k, []).append(v)
    wall = time.perf_counter() - t0
    out["decode_excluded"] = {"fps": round(len(frames) / wall, 1), **stats(totals)}
    out["parts_median_ms"] = {k: round(float(np.median(v)), 2) for k, v in parts.items()}
    del frames
    tracker.reset()
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    per, count = [], 0
    t0 = time.perf_counter()
    while count < n:
        t = time.perf_counter()
        ok, f = cap.read()
        if not ok:
            break
        tracker.process(f, count / fps)
        per.append((time.perf_counter() - t) * 1000)
        count += 1
    wall = time.perf_counter() - t0
    cap.release()
    out["decode_included"] = {"fps": round(count / wall, 1), **stats(per)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pt", default=None, help="also time the .pt via torch (mps, cpu)")
    ap.add_argument("--units", nargs="+", default=list(COMPUTE_UNITS))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--e2e-model", required=True)
    ap.add_argument("--e2e-units", default="ALL")
    ap.add_argument("--video", action="append", default=[], help="label:path[:mirrored]")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"model": {}, "e2e": {}}
    for m in a.models:
        for u in a.units:
            t = time.perf_counter()
            b = load_backend(m, compute_units=u)
            load_s = time.perf_counter() - t
            r = {**stats(b.latency_ms(a.iters, 20)), "load_s": round(load_s, 2)}
            res["model"][f"{Path(m).stem}|{u}"] = r
            print(Path(m).stem, u, r, flush=True)
            del b
    if a.pt:
        for dev in ("mps", "cpu"):
            b = load_backend(a.pt, device=dev)
            r = stats(b.latency_ms(min(a.iters, 100), 10))
            res["model"][f"torch|{dev}"] = r
            print("torch", dev, r, flush=True)
    for spec in a.video:
        label, path, *rest = spec.split(":")
        r = e2e(a.e2e_model, path, bool(rest and rest[0] == "mirrored"), a.frames, a.e2e_units)
        res["e2e"][label] = r
        print(label, json.dumps(r), flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
