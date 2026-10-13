"""Actual alternating inference/flow on contiguous video; model agreement is not GT accuracy."""

# ruff: noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from speedup_tracking import predict, propagate

from gro.infer import load_backend


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--tracking-selection", type=Path, required=True)
    ap.add_argument("--clip", action="append", required=True, help="path:mirrored(0/1):start_seconds")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("Refusing overwrite")
    cv2.setNumThreads(1)
    backend = load_backend(a.model, compute_units="CPU_AND_NE")
    threshold = json.loads(a.selection.read_text())["chosen_threshold"]
    tracking = json.loads(a.tracking_selection.read_text())
    window, flow_size = tracking["window"], tracking.get("flow_size", 0)
    backend.warmup(20)
    report = dict(
        model=str(a.model),
        threshold=threshold,
        window=window,
        clips=[],
        accuracy_note="No dense ground truth. Displacement is agreement with independent model inference, "
        "not accuracy; labelled adjacent-pair accuracy is in the separate tracking report.",
    )
    for spec in a.clip:
        path, mirrored, start = spec.rsplit(":", 2)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_index = round(float(start) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        frames = []
        for _ in range(a.frames):
            ok, im = cap.read()
            if not ok:
                break
            frames.append(cv2.flip(im, 1) if mirrored == "1" else im)
        cap.release()
        assert len(frames) >= 2, path
        elapsed = []
        predictions = []
        failures = 0
        for i, image in enumerate(frames):
            t = time.perf_counter()
            if i % 2 == 0:
                k, _ = predict(backend, image, threshold)
            else:
                k, ok = propagate(frames[i - 1], image, k, window, flow_size)
                failures += int(not ok)
            elapsed.append((time.perf_counter() - t) * 1000)
            predictions.append(k.copy())
        agreement = []
        for i in range(1, len(frames), 2):
            direct, _ = predict(backend, frames[i], threshold)
            valid = (direct[:, 2] > 0.5) & (predictions[i][:, 2] > 0.5)
            agreement.extend(np.linalg.norm(direct[valid, :2] - predictions[i][valid, :2], axis=1).tolist())
        pairs = np.array(elapsed[: len(elapsed) // 2 * 2]).reshape(-1, 2).mean(axis=1)
        report["clips"].append(
            dict(
                path=path,
                start_frame=start_index,
                fps=fps,
                frames=len(frames),
                effective_mean_ms=float(np.mean(elapsed)),
                pair_median_ms=float(np.median(pairs)),
                pair_p95_ms=float(np.percentile(pairs, 95)),
                flow_failures=failures,
                model_agreement_mean_px=float(np.mean(agreement)) if agreement else None,
                frame_ms=elapsed,
            )
        )
        a.out.write_text(json.dumps(report, indent=1))
        print(path, report["clips"][-1]["pair_median_ms"], flush=True)


if __name__ == "__main__":
    main()
