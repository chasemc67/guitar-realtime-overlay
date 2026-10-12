"""CoreML vs PyTorch parity on frames sampled from demo clips (run on a Mac with coremltools + ultralytics).

    python scripts/coreml_parity.py --pt best.pt --coreml v1_fp16.mlpackage v1_int8.mlpackage \
        --clip videos/2026-09-06_09-00-01.mp4:mirrored:24 --clip videos/stan_june2.mp4:mirrored:22 \
        --clip videos/yt_nBpe2YQEzZo.mp4::22 --frames 70 --out parity.json

References: "torch" = the .pt as a plain nn.Module on the same 640x640 letterbox (conversion error only), and
"ultralytics" = YOLO.predict (rect letterbox, e.g. 640x384 for 16:9: the path demo v2 used). Per candidate vs
reference: keypoint px diff (mean / p95 / max over slots with conf > 0.5 in both), |conf diff|, anchor-set flips at
the 0.5 threshold, and wire-midpoint diff (px) between the per-frame fits (no smoothing, known fret count).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gro.infer import FretboardTracker, load_backend  # noqa: E402
from gro.infer.tracker import anchors_from, safe_fit  # noqa: E402


def sample_frames(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append((int(i), f))
    cap.release()
    return out


def ultralytics_kpts(model, canon):
    r = model.predict(canon, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.full((50, 3), np.nan, np.float32)
    k = int(r.boxes.conf.argmax().item())
    return np.concatenate([r.keypoints.xy[k].numpy(), r.keypoints.conf[k].numpy()[:, None]], 1).astype(np.float32)


def wires(k, frets):
    fit = safe_fit([p for p in anchors_from(k) if p.fret <= frets], frets)
    if fit is None:
        return None
    return np.array([fit.project_fy(n, 0.5) for n in range(frets + 1)])


def compare(a, b, frets):
    """Per-frame stats of candidate a vs reference b ((50, 3) each)."""
    both = (a[:, 2] > 0.5) & (b[:, 2] > 0.5)
    either = (a[:, 2] > 0.5) | (b[:, 2] > 0.5)
    d = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    wa, wb = wires(a, frets), wires(b, frets)
    return {
        "px": d[both].tolist(),
        "conf": np.abs(a[:, 2] - b[:, 2])[np.isfinite(a[:, 2]) & np.isfinite(b[:, 2])].tolist(),
        "flips": int((either & ~both).sum()),
        "slots": int(both.sum()),
        "wire_px": None if wa is None or wb is None else np.linalg.norm(wa - wb, axis=1).tolist(),
        "fit_lost": int((wa is None) != (wb is None)),
    }


def summarize(rows):
    px = np.concatenate([r["px"] for r in rows]) if rows else np.zeros(0)
    conf = np.concatenate([r["conf"] for r in rows])
    wire = [w for r in rows if r["wire_px"] is not None for w in r["wire_px"]]
    wire_max_per_frame = [max(r["wire_px"]) for r in rows if r["wire_px"] is not None]
    f = lambda x, q: round(float(np.percentile(x, q)), 3) if len(x) else None  # noqa: E731
    return {
        "frames": len(rows),
        "kpt_px_mean": round(float(px.mean()), 3) if len(px) else None,
        "kpt_px_p95": f(px, 95),
        "kpt_px_max": round(float(px.max()), 3) if len(px) else None,
        "conf_mean": round(float(conf.mean()), 4),
        "conf_max": round(float(conf.max()), 4),
        "anchor_flips_per_frame": round(sum(r["flips"] for r in rows) / len(rows), 3),
        "anchors_per_frame": round(sum(r["slots"] for r in rows) / len(rows), 2),
        "wire_px_mean": round(float(np.mean(wire)), 3) if wire else None,
        "wire_px_p95": f(wire, 95),
        "wire_px_max": round(float(np.max(wire)), 3) if wire else None,
        "frame_wire_max_median": f(wire_max_per_frame, 50),
        "fit_lost_frames": sum(r["fit_lost"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--coreml", nargs="+", required=True)
    ap.add_argument("--clip", action="append", required=True, help="path[:mirrored][:frets]")
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--compute-units", default="ALL")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from ultralytics import YOLO

    yolo = YOLO(a.pt, task="pose")
    trackers = {"torch": FretboardTracker(load_backend(a.pt, device="cpu"))}
    for p in a.coreml:
        trackers[Path(p).stem] = FretboardTracker(load_backend(p, compute_units=a.compute_units))
    pairs = [(c, "torch") for c in trackers if c != "torch"] + [(c, "ultralytics") for c in trackers]
    rows = {f"{c}|{r}": {} for c, r in pairs}
    for spec in a.clip:
        path, mirrored, frets = (spec.split(":") + ["", ""])[:3]
        mirrored, frets = mirrored == "mirrored", int(frets or 24)
        name = Path(path).stem
        for _, frame in sample_frames(path, a.frames):
            canon = cv2.flip(frame, 1) if mirrored else frame
            k = {n: t.infer(canon)[0] for n, t in trackers.items()}
            k["ultralytics"] = ultralytics_kpts(yolo, canon)
            for c, r in pairs:
                rows[f"{c}|{r}"].setdefault(name, []).append(compare(k[c], k[r], frets))
        print(name, "done", flush=True)
    report = {}
    for key, clips in rows.items():
        report[key] = {name: summarize(v) for name, v in clips.items()}
        report[key]["all"] = summarize([r for v in clips.values() for r in v])
    a.out.write_text(json.dumps(report, indent=1))
    for key, v in report.items():
        print(key, json.dumps(v["all"]))


if __name__ == "__main__":
    main()
