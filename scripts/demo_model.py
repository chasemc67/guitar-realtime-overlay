"""Render a smoothed model overlay: --weights best.pt --video clip.mp4 --out overlay.mp4.

Requires ultralytics and imageio-ffmpeg (not needed with --preds from scripts/cache_preds.py). The D1 mirrored flag
and a known fret count come from data_manifests/ (gro.auto.manifest.clip_info); --[no-]mirrored / --frets override.
Board end: the last fret is detected per frame (gro.board_end: keypoint confidence + fret-wire ridges in the rectified
strip) and voted per clip; the grid and notes stop there. Smoothing: One-Euro on the homography (gro.smooth), or the
v1 anchor EMA with --smooth ema. An unusable detection holds the last fit for at most 10 frames.
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

from gro.anchors import Anchor  # noqa: E402
from gro.auto.manifest import clip_info  # noqa: E402
from gro.board_end import FretCountVote, frame_last_fret, ridge_scores  # noqa: E402
from gro.geometry import FitError, fit_fretboard  # noqa: E402
from gro.render import render_on_original  # noqa: E402
from gro.smooth import OneEuroModel  # noqa: E402
from gro.theory import scale_positions  # noqa: E402

END_FIT_MAX_FRET = 21  # board-end probe fit: ignore slots that may sit on the pickups of a 21/22-fret guitar


def anchors_from(k, thr=0.5, max_fret=24):
    """Anchors from one (50, 3) x, y, conf keypoint array (absolute slots 0b, 0t, ..., 24t)."""
    return [
        Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
        for i, (x, y, c) in enumerate(k)
        if np.isfinite(c) and c > thr and i // 2 <= max_fret
    ]


def safe_fit(anchors, n_frets=24):
    try:
        m = fit_fretboard(anchors, n_frets=n_frets)
    except (FitError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return m if np.isfinite(m.H).all() and np.isfinite(m.rmse_px) else None


class SmoothTracker:
    """Per-frame fit on anchors up to the board end, One-Euro smoothed; holds the last fit for `hold` frames."""

    def __init__(self, fps, smooth=True, hold=10, **one_euro):
        self.filter = OneEuroModel(fps, **one_euro) if smooth else None
        self.hold, self.missed, self.model = hold, 0, None

    def update(self, anchors, shape, n_frets=24):
        fit = safe_fit([p for p in anchors if p.fret <= n_frets], n_frets)
        if fit is None:
            self.missed += 1
            if self.missed > self.hold:
                self.model = None
                if self.filter:
                    self.filter.reset()
            return self.model
        self.missed = 0
        self.model = self.filter.update(fit, shape) if self.filter else fit
        return self.model


class EmaTracker:
    """v1 smoothing (kept for comparison): EMA on anchor coordinates, always 24 frets."""

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
    ap.add_argument("--weights", default=None, help="pose weights (not needed with --preds)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preds", type=Path, default=None, help="prediction cache from scripts/cache_preds.py")
    ap.add_argument("--scale", default="A minor pentatonic")
    ap.add_argument("--mirrored", action=argparse.BooleanOptionalAction, default=None, help="default: manifest (D1)")
    ap.add_argument("--frets", type=int, default=None, help="fret count override (default: manifest, else detect)")
    ap.add_argument("--smooth", choices=("one_euro", "ema", "none"), default="one_euro")
    ap.add_argument("--min-cutoff", type=float, default=0.4, help="One-Euro min cutoff, Hz")
    ap.add_argument("--beta", type=float, default=3.0, help="One-Euro speed coefficient (per fret width/s)")
    ap.add_argument("--speed-floor", type=float, default=2.0, help="One-Euro noise speed floor, fret widths/s")
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
    if a.preds is None and a.weights is None:
        ap.error("need --weights or --preds")
    try:
        root, scale = a.scale.split(maxsplit=1)
        notes = scale_positions(root, scale)
    except (ValueError, KeyError) as exc:
        ap.error(str(exc))
    info = clip_info(a.video)
    mirrored = info["mirrored"] if a.mirrored is None else a.mirrored
    override = a.frets or info["frets"]
    cache = None
    if a.preds is not None:
        cache = np.load(a.preds)
        if bool(cache["mirrored"]) != mirrored:
            ap.error(f"--preds was cached with mirrored={bool(cache['mirrored'])}, this run uses {mirrored}")
        cache = dict(zip(cache["frame"].tolist(), cache["kpts"], strict=True))
    else:
        from cache_preds import raw_predict
        from ultralytics import YOLO

        model = YOLO(a.weights, task="pose")
    import imageio_ffmpeg

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
            str(a.out),
            (w, h),
            fps=fps / a.every,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", str(a.crf), "-preset", "medium", "-movflags", "+faststart", *scale],
        )
        writer.send(None)
        if a.smooth == "ema":
            tracker = EmaTracker()
        else:
            tracker = SmoothTracker(
                fps,
                smooth=a.smooth == "one_euro",
                min_cutoff=a.min_cutoff,
                beta=a.beta,
                speed_floor=a.speed_floor,
            )
        vote = FretCountVote(override=override)
        while ok and (not a.max_frames or count < a.max_frames):
            canon = cv2.flip(frame, 1) if mirrored else frame
            if cache is not None:
                if index not in cache:
                    break
                k = cache[index]
            else:
                k, _ = raw_predict(model, canon, a.imgsz, a.device)
            anchors = anchors_from(k)
            estimate = None
            if not override:
                probe = safe_fit([p for p in anchors if p.fret <= END_FIT_MAX_FRET])
                if probe is not None:
                    conf = np.nanmin(k[:, 2].reshape(25, 2), 1)
                    estimate = frame_last_fret(ridge_scores(canon, probe), conf)
            n_frets = vote.update(estimate)
            fit = tracker.update(anchors) if a.smooth == "ema" else tracker.update(anchors, canon.shape, n_frets)
            out = render_on_original(frame, fit, mirrored, notes=notes) if fit is not None else frame
            if count % a.every == 0:
                writer.send(np.ascontiguousarray(out[:, :, ::-1]))
            frets = sorted({p.fret for p in anchors})
            log.append(
                {
                    "frame": index,
                    "t": round(index / fps, 3),
                    "n_anchors": len(anchors),
                    "fret_lo": frets[0] if frets else "",
                    "fret_hi": frets[-1] if frets else "",
                    "end_estimate": estimate if estimate is not None else "",
                    "n_frets": n_frets,
                    "missed": tracker.missed,
                    "has_fit": int(fit is not None),
                    "rmse_px": round(fit.rmse_px, 3) if fit is not None else "",
                    "inliers": len(fit.inliers) if fit is not None else "",
                }
            )
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
    votes = dict(sorted(vote.votes.items()))
    print(
        f"{a.out}: {count} frames; mirrored={mirrored}; frets={vote.value} "
        f"({'override' if override else f'detected, per-frame votes {votes}'}); "
        f"avg {count / (time.perf_counter() - start):.2f} fps"
    )


if __name__ == "__main__":
    main()
