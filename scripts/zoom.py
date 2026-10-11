"""Annotation aid: crop a frame region, upscale it, and draw a labelled pixel grid (full-frame coordinates).

uv run python scripts/zoom.py FRAME X0 Y0 X1 Y1 [--scale 4] [--step 10] [--out PATH] [--anchors JSON]
"""

from __future__ import annotations

import argparse

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("box", type=int, nargs=4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--mirror", action="store_true", help="mirror the frame first (canonical coordinates)")
    ap.add_argument("--points", default="", help="x,y;x,y;... to mark (full-frame coords)")
    ap.add_argument("--out", default="/tmp/zoom.png")
    a = ap.parse_args()
    img = cv2.imread(a.frame)
    if a.mirror:
        img = cv2.flip(img, 1)
    x0, y0, x1, y1 = a.box
    s = a.scale
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for x in range((x0 // a.step + 1) * a.step, x1, a.step):
        major = x % (a.step * 5) == 0
        cv2.line(
            crop,
            ((x - x0) * s, 0),
            ((x - x0) * s, crop.shape[0]),
            (0, 255, 255) if major else (0, 140, 140),
            1,
        )
        if major:
            cv2.putText(crop, str(x), ((x - x0) * s + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range((y0 // a.step + 1) * a.step, y1, a.step):
        major = y % (a.step * 5) == 0
        cv2.line(
            crop,
            (0, (y - y0) * s),
            (crop.shape[1], (y - y0) * s),
            (255, 0, 255) if major else (140, 0, 140),
            1,
        )
        if major:
            cv2.putText(crop, str(y), (2, (y - y0) * s - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    if a.points:
        for p in a.points.split(";"):
            px, py = map(float, p.split(","))
            c = (int(round((px - x0 + 0.5) * s)), int(round((py - y0 + 0.5) * s)))
            cv2.circle(crop, c, 5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, crop)
    print(a.out, crop.shape)


if __name__ == "__main__":
    main()
