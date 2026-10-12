"""Cache raw model keypoints per frame: --weights best.pt --video clip.mp4 --out preds.npz [--start s --seconds s].

Inference runs on the canonical frame (D1: h-flipped when the manifest/`--mirrored` says so). The cache lets
demo_model.py (`--preds`) and jitter_eval.py iterate on smoothing / board-end logic without re-running the network.
Arrays: kpts (T, 50, 3) canonical x, y, conf (NaN when no board), box_conf (T,), frame (T,) source frame index.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro.auto.manifest import clip_info  # noqa: E402


def raw_predict(model, image, imgsz=640, device=None):
    """(50, 3) x, y, conf of the highest-confidence board (NaN if none), board confidence."""
    result = model.predict(image, imgsz=imgsz, device=device, conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.full((50, 3), np.nan, np.float32), 0.0
    k = int(result.boxes.conf.argmax().item())
    xy = result.keypoints.xy[k].cpu().numpy()
    conf = result.keypoints.conf[k].cpu().numpy()
    return np.concatenate([xy, conf[:, None]], 1).astype(np.float32), float(result.boxes.conf[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mirrored", action=argparse.BooleanOptionalAction, default=None, help="default: manifest")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    mirrored = clip_info(a.video)["mirrored"] if a.mirrored is None else a.mirrored
    from ultralytics import YOLO

    model = YOLO(a.weights, task="pose")
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if a.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    last = index + int(round(a.seconds * fps)) if a.seconds else 1 << 30
    kpts, box, frames, size = [], [], [], (0, 0)
    t0 = time.perf_counter()
    while index < last:
        ok, frame = cap.read()
        if not ok:
            break
        size = frame.shape[1::-1]
        k, b = raw_predict(model, cv2.flip(frame, 1) if mirrored else frame, a.imgsz, a.device)
        kpts.append(k), box.append(b), frames.append(index)
        index += 1
    cap.release()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        kpts=np.array(kpts),
        box_conf=np.array(box, np.float32),
        frame=np.array(frames),
        fps=fps,
        size=np.array(size),
        mirrored=mirrored,
        video=str(a.video),
        imgsz=a.imgsz,
    )
    print(f"{a.out}: {len(kpts)} frames, mirrored={mirrored}, {len(kpts) / (time.perf_counter() - t0):.1f} fps")


if __name__ == "__main__":
    main()
