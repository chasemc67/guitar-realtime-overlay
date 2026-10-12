"""Render a smoothed model overlay: --weights best.pt --video clip.mp4 --out overlay.mp4.

Requires ultralytics and imageio-ffmpeg. --mirrored canonicalizes inference and preserves readable overlay text.
EMA is applied to anchor coordinates; an unusable detection holds the last fit for at most 10 frames.
Output is H.264 (libx264, --crf, scaled to --width); --log writes a per-frame CSV (anchors, fit rmse, held/missed).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_model import predict  # noqa: E402

from gro.anchors import Anchor  # noqa: E402
from gro.geometry import FitError, fit_fretboard  # noqa: E402
from gro.render import render_on_original  # noqa: E402
from gro.theory import scale_positions  # noqa: E402


class Tracker:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.model = None
        self.previous = {}
        self.missed = 0

    def update(self, anchors):
        smoothed = []
        for p in anchors:
            old = self.previous.get((p.fret, p.side), np.array([p.x, p.y]))
            xy = self.alpha * np.array([p.x, p.y]) + (1 - self.alpha) * old
            smoothed.append(Anchor(p.fret, p.side, *xy, conf=p.conf))
        try:
            model = fit_fretboard(smoothed)
            if not np.isfinite(model.H).all() or not np.isfinite(model.rmse_px):
                raise FitError("non-finite fit")
        except (FitError, np.linalg.LinAlgError, FloatingPointError):
            self.missed += 1
            if self.missed > 10:
                self.model, self.previous = None, {}
        else:
            self.model, self.missed = model, 0
            self.previous = {(p.fret, p.side): np.array([p.x, p.y]) for p in model.inliers}
        return self.model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", default="A minor pentatonic")
    ap.add_argument("--mirrored", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=960, help="output width (0 = source)")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--every", type=int, default=1, help="write every n-th frame (tracking still sees all)")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()
    if a.out.resolve() == a.video.resolve() or a.out.suffix.lower() != ".mp4":
        ap.error("--out must be an MP4 distinct from the source")
    try:
        root, scale = a.scale.split(maxsplit=1)
        notes = scale_positions(root, scale)
    except (ValueError, KeyError) as exc:
        ap.error(str(exc))
    from ultralytics import YOLO

    import imageio_ffmpeg

    model = YOLO(a.weights, task="pose")
    cap = cv2.VideoCapture(str(a.video))
    writer = None
    log = []
    count = 0
    start = time.perf_counter()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if a.start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read()
        if not ok or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"cannot read video/source fps: {a.video}")
        h, w = frame.shape[:2]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        scale = ["-vf", f"scale={a.width}:-2"] if a.width and a.width != w else []
        writer = imageio_ffmpeg.write_frames(
            str(a.out), (w, h), fps=fps / a.every, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
            output_params=["-crf", str(a.crf), "-preset", "medium", "-movflags", "+faststart", *scale],
        )
        writer.send(None)
        tracker = Tracker()
        while ok and (not a.max_frames or count < a.max_frames):
            canon = cv2.flip(frame, 1) if a.mirrored else frame
            anchors, detected = predict(model, canon, a.imgsz, a.device)
            fit = tracker.update(anchors)
            out = render_on_original(frame, fit, a.mirrored, notes=notes) if fit is not None else frame
            if count % a.every == 0:
                writer.send(np.ascontiguousarray(out[:, :, ::-1]))
            frets = sorted({p.fret for p in anchors})
            log.append({
                "frame": index, "t": round(index / fps, 3), "detected": int(detected), "n_anchors": len(anchors),
                "fret_lo": frets[0] if frets else "", "fret_hi": frets[-1] if frets else "",
                "missed": tracker.missed, "has_fit": int(fit is not None),
                "rmse_px": round(fit.rmse_px, 3) if fit is not None else "",
                "inliers": len(fit.inliers) if fit is not None else "",
            })
            count += 1
            index += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if a.log is not None and log:
        with a.log.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0]))
            wr.writeheader()
            wr.writerows(log)
    print(f"{a.out}: {count} frames; avg {count / (time.perf_counter() - start):.2f} fps (decode/infer/render/write)")


if __name__ == "__main__":
    main()
