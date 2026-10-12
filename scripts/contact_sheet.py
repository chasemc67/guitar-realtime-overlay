"""Contact sheet of a rendered video: --video demo.mp4 --out sheet.jpg [--spread 16] [--runs 2] [--run-len 5]
[--frames-dir dir]. `spread` frames evenly over the clip, plus `runs` runs of `run-len` consecutive frames (for jitter);
tiles are captioned with the frame index. --frames-dir also writes each tile at full output size."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spread", type=int, default=16)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--run-len", type=int, default=5)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-w", type=int, default=384)
    ap.add_argument("--frames-dir", type=Path)
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(a.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = list(np.linspace(0, n - 1, a.spread + 2)[1:-1].astype(int))
    for r in range(a.runs):
        s = int(n * (r + 1) / (a.runs + 1) + n * 0.05)
        picks += list(range(s, min(s + a.run_len, n)))
    tiles = []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        if a.frames_dir:
            a.frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(a.frames_dir / f"{a.video.stem}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t = cv2.resize(img, (a.tile_w, int(img.shape[0] * a.tile_w / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"#{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, f"#{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + a.cols]) for i in range(0, len(tiles), a.cols)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out, len(picks), "frames")


if __name__ == "__main__":
    main()
