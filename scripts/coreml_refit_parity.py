"""CoreML fp16 parity before/after the frozen 0.30-fret refit; latency ALL/CPU_AND_NE."""

# ruff: noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from coreml_parity import compare, sample_frames, summarize

from gro.fretnet.numbering import anchors_from_k, reassign_wires
from gro.geometry import FitError, fit_fretboard
from gro.infer import Letterbox, decode, load_backend


def refit(k):
    try:
        fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
        return reassign_wires(k, fit, tolerance=0.30, reject_ambiguous=True)
    except (FitError, ValueError, np.linalg.LinAlgError):
        return k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--coreml", required=True)
    ap.add_argument("--clip", action="append", required=True, help="path:mirrored:frets")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pt = load_backend(a.pt, device="cpu")
    report = {"threshold": 0.30, "latency": {}, "parity": {}}
    for units in ("ALL", "CPU_AND_NE"):
        started = time.perf_counter()
        ml = load_backend(a.coreml, compute_units=units)
        load_seconds = time.perf_counter() - started
        assert ml.imgsz == pt.imgsz == 832
        ms = ml.latency_ms(200, 20)
        report["latency"][units] = dict(
            median_ms=float(np.median(ms)),
            p95_ms=float(np.percentile(ms, 95)),
            load_seconds=load_seconds,
            calls=200,
            warmups=20,
        )
        print(units, report["latency"][units], flush=True)
        # Same 90 source frames as the previous v3 parity protocol, 30 per clip.
        rows = {"raw": [], "refit": []}
        for spec in a.clip:
            path, mirrored, frets = spec.split(":")
            for index, img in sample_frames(path, a.frames):
                if mirrored == "mirrored":
                    img = cv2.flip(img, 1)
                lb = Letterbox.for_shape(img.shape, 832)
                rgb = cv2.cvtColor(lb.apply(img), cv2.COLOR_BGR2RGB)
                p = decode(pt(rgb))[0]
                m = decode(ml(rgb))[0]
                for name, pk, mk in [("raw", p.copy(), m.copy()), ("refit", refit(p), refit(m))]:
                    pk[:, :2] = lb.to_frame(pk[:, :2])
                    mk[:, :2] = lb.to_frame(mk[:, :2])
                    rows[name].append(dict(clip=Path(path).stem, frame=index, **compare(mk, pk, int(frets))))
            print(units, Path(path).name, "done", flush=True)
        report["parity"][units] = {name: summarize(rs) for name, rs in rows.items()}
        report.setdefault("frames", {})[units] = rows
        a.out.write_text(json.dumps(report, indent=1) + "\n")
        del ml
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
