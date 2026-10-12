"""Startup fret count before/after: replay cached predictions through gro.infer.FretboardTracker for the first second.

"before" = vote only (24 frets drawn until the board-end vote switches); "after" = capped at the highest confident
fret until the vote settles. Writes one still (rows = clips, columns = before / after at --at seconds) and prints the
drawn fret count per frame.

    python scripts/startup_before_after.py --clip 0906 work/preds_0906.npz data/raw/2026-09-06_09-00-01.mp4 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro.infer import FretboardTracker, render_state  # noqa: E402
from gro.infer.backends import Backend  # noqa: E402
from gro.theory import scale_positions  # noqa: E402


class Replay(Backend):
    name = "replay"


def run(preds: Path, video: Path, seconds: float, cap: bool, at: float):
    d = np.load(preds)
    kpts = dict(zip(d["frame"].tolist(), d["kpts"], strict=True))
    fps = float(d["fps"])
    tr = FretboardTracker(Replay(), mirrored=bool(d["mirrored"]), fps=fps, end_every=1)
    if not cap:
        tr.vote.observe_conf = lambda conf: None
    notes = scale_positions("A", "minor pentatonic")
    cap_ = cv2.VideoCapture(str(video))
    counts, still = [], None
    for i in range(int(round(seconds * fps))):
        ok, frame = cap_.read()
        if not ok:
            break
        canon = tr.canonical(frame)
        k = kpts[i].copy()
        st = tr.update(canon, k, i / fps)
        counts.append(st.n_frets)
        if still is None and i >= at * fps:
            still = render_state(frame, st, notes=notes)
            cv2.putText(
                still,
                f"{'after' if cap else 'before'}: t={i / fps:.2f}s, {st.n_frets} frets drawn",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 0),
                8,
                cv2.LINE_AA,
            )
            cv2.putText(
                still,
                f"{'after' if cap else 'before'}: t={i / fps:.2f}s, {st.n_frets} frets drawn",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cap_.release()
    return counts, still


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", nargs=3, action="append", metavar=("NAME", "PREDS", "VIDEO"), required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--at", type=float, default=0.2, help="still time, seconds")
    ap.add_argument("--width", type=int, default=800, help="tile width")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for name, preds, video in a.clip:
        tiles = []
        for cap in (False, True):
            counts, still = run(Path(preds), Path(video), a.seconds, cap, a.at)
            runs = []
            for c in counts:
                if runs and runs[-1][0] == c:
                    runs[-1][1] += 1
                else:
                    runs.append([c, 1])
            print(
                f"{name} {'after ' if cap else 'before'}: frets drawn per frame (value x frames) "
                + " ".join(f"{c}x{n}" for c, n in runs)
            )
            h, w = still.shape[:2]
            tiles.append(cv2.resize(still, (a.width, round(h * a.width / w)), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out)


if __name__ == "__main__":
    main()
