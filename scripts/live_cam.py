"""Live / real-time-paced fretboard overlay with FretboardTracker.

    python scripts/live_cam.py --model v1_fp16.mlpackage --source 0 --show            # webcam 0
    python scripts/live_cam.py --model v1_fp16.mlpackage --source clip.mp4 --mirrored  # video at real-time pace
    python scripts/live_cam.py --model ... --source clip.mp4 --offline --out out.mp4   # every frame, render to H.264

Real-time mode: a reader thread produces frames at the source rate (a camera, or a video paced by its timestamps)
into a one-slot buffer; the tracker always takes the newest frame and older ones are dropped, like a live camera.
Reports achieved fps, dropped frames and latency (frame capture -> tracker output). --out needs imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gro.infer import FretboardTracker, draw_hud, render_state  # noqa: E402
from gro.theory import scale_positions  # noqa: E402


class LatestFrame:
    """Reader thread: keeps only the newest (frame, capture_time, index); counts frames overwritten unread.

    realtime=False (video only): no pacing and no drops, the reader waits for the consumer."""

    def __init__(self, source, realtime: bool, start_s: float = 0.0, seconds: float = 0.0):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        self.is_camera = str(source).isdigit()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if start_s and not self.is_camera:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        self.limit = int(seconds * self.fps) if seconds else 0
        self.realtime = realtime or self.is_camera
        self.cond = threading.Condition()
        self.item, self.done, self.produced, self.dropped = None, False, 0, 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.t0 = time.perf_counter()
        self.thread.start()
        return self

    def _run(self):
        while not self.limit or self.produced < self.limit:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.realtime and not self.is_camera:
                due = self.t0 + self.produced / self.fps  # pace the file like a camera
                time.sleep(max(0.0, due - time.perf_counter()))
            with self.cond:
                while not self.realtime and self.item is not None:
                    self.cond.wait()
                if self.item is not None:
                    self.dropped += 1
                self.item = (frame, time.perf_counter(), self.produced)
                self.produced += 1
                self.cond.notify()
        with self.cond:
            self.done = True
            self.cond.notify()

    def get(self):
        with self.cond:
            while self.item is None and not self.done:
                self.cond.wait()
            item, self.item = self.item, None
            self.cond.notify()
            return item


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help=".mlpackage / .onnx / .pt")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--mirrored", action="store_true", help="D1: canonical = horizontally flipped frame")
    ap.add_argument("--frets", type=int, default=None)
    ap.add_argument("--compute-units", default="ALL", help="CoreML: ALL / CPU_AND_NE / CPU_AND_GPU / CPU_ONLY")
    ap.add_argument("--scale", default="A minor pentatonic")
    ap.add_argument("--offline", action="store_true", help="process every frame as fast as possible (no drops)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--render", action="store_true", help="draw the overlay even without --show / --out (timing)")
    ap.add_argument("--hud", action="store_true", help="draw fps / latency on the output")
    ap.add_argument("--out", type=Path, default=None, help="H.264 MP4 of the rendered frames")
    ap.add_argument("--width", type=int, default=960, help="--out width")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--stats", type=Path, default=None, help="write the summary JSON here")
    a = ap.parse_args()
    root, scale = a.scale.split(maxsplit=1)
    notes = scale_positions(root, scale)
    kw = {"compute_units": a.compute_units} if a.model.endswith((".mlpackage", ".mlmodel")) else {}
    reader = LatestFrame(a.source, realtime=not a.offline, start_s=a.start, seconds=a.seconds)
    tracker = FretboardTracker(a.model, mirrored=a.mirrored, frets=a.frets, fps=reader.fps, **kw)
    tracker.backend.warmup()
    writer = None
    lat, proc, valid, n = [], [], 0, 0
    reader.start()
    t_start = time.perf_counter()
    try:
        while True:
            item = reader.get()
            if item is None:
                break
            frame, t_cap, index = item
            state = tracker.process(frame, index / reader.fps)
            done = time.perf_counter()
            lat.append((done - t_cap) * 1000)
            proc.append(state.timings_ms["total"])
            valid += state.valid
            n += 1
            if a.out or a.show or a.render:
                out = render_state(frame, state, notes=notes)
                if a.hud:
                    elapsed = done - t_start
                    draw_hud(out, f"{n / elapsed:4.1f} fps  {lat[-1]:5.1f} ms  frets {state.n_frets}")
                if a.out:
                    if writer is None:
                        import imageio_ffmpeg

                        h, w = out.shape[:2]
                        sc = ["-vf", f"scale={a.width}:-2"] if a.width and a.width != w else []
                        writer = imageio_ffmpeg.write_frames(
                            str(a.out),
                            (w, h),
                            fps=reader.fps,
                            codec="libx264",
                            pix_fmt_out="yuv420p",
                            macro_block_size=1,
                            output_params=["-crf", str(a.crf), "-preset", "medium", "-movflags", "+faststart", *sc],
                        )
                        writer.send(None)
                    writer.send(np.ascontiguousarray(out[:, :, ::-1]))
                if a.show:
                    cv2.imshow("gro", out)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            if n % 100 == 0:
                print(f"{n} frames, {n / (time.perf_counter() - t_start):.1f} fps", flush=True)
    finally:
        if writer is not None:
            writer.close()
        reader.cap.release()
    wall = time.perf_counter() - t_start
    summary = {
        "source": str(a.source),
        "mode": "offline" if a.offline else "realtime",
        "source_fps": round(reader.fps, 2),
        "frames_read": reader.produced,
        "frames_processed": n,
        "dropped": reader.produced - n,
        "achieved_fps": round(n / wall, 1),
        "latency_ms_median": round(float(np.median(lat)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "process_ms_median": round(float(np.median(proc)), 1),
        "process_ms_p95": round(float(np.percentile(proc, 95)), 1),
        "valid_frac": round(valid / max(n, 1), 3),
        "frets": tracker.vote.value,
    }
    print(json.dumps(summary))
    if a.stats:
        a.stats.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
