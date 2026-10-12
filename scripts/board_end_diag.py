"""Board-end diagnostics from a prediction cache: per-frame ridge ratios at wires 19..24, the per-frame last-fret
estimate and the clip vote; writes <out>/board_end_<name>.csv and zoomed high-fret crops + rectified strips for
manual wire counting.  --preds preds.npz --out dir [--every 10] [--crops 6]"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo_model import anchors_from  # noqa: E402

from gro.board_end import FretCountVote, frame_last_fret, rectify_strip, ridge_scores  # noqa: E402
from gro.geometry import FitError, fit_fretboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--crops", type=int, default=6)
    a = ap.parse_args()
    d = np.load(a.preds)
    name = a.preds.stem.removeprefix("preds_")
    cap = cv2.VideoCapture(str(ROOT / str(d["video"])))
    mirrored = bool(d["mirrored"])
    a.out.mkdir(parents=True, exist_ok=True)
    rows, est, vote = [], Counter(), FretCountVote()
    crops = []
    for t, (k, fi) in enumerate(zip(d["kpts"], d["frame"], strict=True)):
        if t % a.every:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            break
        canon = cv2.flip(frame, 1) if mirrored else frame
        try:
            m = fit_fretboard(anchors_from(k, max_fret=21))
        except FitError:
            continue
        r = ridge_scores(canon, m)
        conf = np.nanmin(k[:, 2].reshape(25, 2), 1)
        n = frame_last_fret(r, conf)
        v = vote.update(n)
        est[n] += 1
        rows.append(
            {
                "frame": int(fi),
                "est": n,
                "vote": v,
                **{f"r{f}": round(r.get(f, np.nan), 2) for f in range(19, 25)},
                **{f"c{f}": round(float(conf[f]), 2) for f in range(19, 25)},
            }
        )
        if r and len(r) == 6:
            crops.append((int(fi), canon, m, r, n))
    with (a.out / f"board_end_{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(name, "per-frame estimates:", dict(est), "final vote:", vote.value)
    for fi, canon, m, r, n in [crops[i] for i in np.linspace(0, len(crops) - 1, min(a.crops, len(crops))).astype(int)]:
        # zoomed crop around frets 16..26 (canonical image), wires 19..24 ticked at the predicted positions
        pts = np.concatenate([m.fret_line(f) for f in np.arange(16, 26.5, 0.5)])
        x0, y0 = np.maximum(pts.min(0) - 30, 0).astype(int)
        x1, y1 = np.minimum(pts.max(0) + 30, canon.shape[1::-1]).astype(int)
        crop = canon[y0:y1, x0:x1].copy()
        for f in range(16, 25):
            p = m.fret_line(f) - [x0, y0]
            cv2.putText(
                crop, str(f), tuple(int(v) for v in p[0] + [-6, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
            )
        s = max(1, int(700 / max(crop.shape[1], 1)))
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        strip, _, _ = rectify_strip(canon, m, 13.5, 24.5)
        strip = cv2.resize(cv2.cvtColor(strip.astype(np.uint8), cv2.COLOR_GRAY2BGR), (crop.shape[1], 80))
        txt = f"{name} f{fi} est={n} " + " ".join(f"{f}:{r[f]:.2f}" for f in range(19, 25))
        bar = np.zeros((24, crop.shape[1], 3), np.uint8)
        cv2.putText(bar, txt, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(a.out / f"end_{name}_{fi}.jpg"), np.vstack([bar, crop, strip]))


if __name__ == "__main__":
    main()
