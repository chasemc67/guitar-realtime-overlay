"""Re-derive GT anchors under PLAN D13 (anchors = fret wire x outermost strings, low E / high E).

    uv run python scripts/rederive_gt_strings.py samples/anchors/*.json samples/anchors_extra/*.json [--crops DIR]

Each GT wire line (the snapped, hand-read wire through its old bass/treble anchors) is kept; only the position along
the wire moves. The string comb (`gro.auto.strings`) is found on the board of a model fitted to the GT anchors, then
re-found on the string-convention refit (2 rounds). The old edge-convention anchors move to the frame's optional
`board_edge_kpts` field. `--crops` writes zoom crops (old anchor = green circle, detected strings = dashed, outer
strings red) for checking by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.detect import prep_gray
from gro.auto.strings import detect_strings
from gro.geometry import fit_fretboard
from gro.snap import intersect

ROOT = Path(__file__).resolve().parents[1]


def _crops(canon, anchors, sf, frets, z: int = 6, h: int = 24) -> np.ndarray:
    rows = []
    for side in A.SIDES:
        tiles = []
        for f in frets[1 :: max(1, len(frets) // 6)][:6]:
            a = next(x for x in anchors if x.fret == f and x.side == side)
            cx, cy = int(a.x), int(a.y)
            c = cv2.copyMakeBorder(canon, h, h, h, h, cv2.BORDER_CONSTANT)[cy : cy + 2 * h, cx : cx + 2 * h]
            t = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
            for i, (p, u) in sf.image_lines.items():
                q0, q1 = (p - 1000 * u - [cx - h, cy - h]) * z, (p + 1000 * u - [cx - h, cy - h]) * z
                col = (0, 0, 255) if i in (0, 5) else (255, 160, 0)
                for k in range(0, 200, 2):
                    a0, a1 = q0 + (q1 - q0) * k / 200, q0 + (q1 - q0) * (k + 1) / 200
                    cv2.line(t, tuple(map(int, a0)), tuple(map(int, a1)), col, 1)
            cv2.circle(t, (int((a.x - cx + h + 0.5) * z), int((a.y - cy + h + 0.5) * z)), 7, (0, 255, 0), 1)
            cv2.putText(t, f"{f}{side[0]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(np.pad(t, ((1, 1), (1, 1), (0, 0))))
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])


def rederive(path: Path, crops: Path | None) -> None:
    lf = A.load(path)
    for i, fr in enumerate(lf.frames):
        if not fr.present() or "board_edge_kpts" in fr.extra:
            continue
        img = cv2.imread(str(ROOT / fr.extra["frame_file"]))
        canon = cv2.flip(img, 1) if lf.mirrored else img
        gray = prep_gray(canon)
        old = [a for a in fr.anchors if a.vis > 0]
        by = {}
        for a in old:
            by.setdefault(a.fret, {})[a.side] = a
        wires = {f: d for f, d in by.items() if len(d) == 2}
        vis2 = sorted(f for f, d in wires.items() if f > 0 and d["bass"].vis == 2 and d["treble"].vis == 2)
        anchors = old
        moves = []
        for _ in range(2):
            model = fit_fretboard([a for a in anchors if a.vis == 2], tau=None)
            sf = detect_strings(gray, model, vis2)
            new = []
            for f, d in sorted(wires.items()):
                b, t = np.array([d["bass"].x, d["bass"].y]), np.array([d["treble"].x, d["treble"].y])
                wl = (b, (t - b) / np.linalg.norm(t - b))
                for side, ln in zip(A.SIDES, sf.lines, strict=True):
                    p = intersect(wl, ln)
                    new.append(A.Anchor(f, side, round(float(p[0]), 2), round(float(p[1]), 2), d[side].vis))
            anchors = new
        om = {(a.fret, a.side): a for a in old}
        moves = {
            s: [float(np.hypot(a.x - om[a.fret, a.side].x, a.y - om[a.fret, a.side].y)) for a in anchors if a.side == s]
            for s in A.SIDES
        }
        print(
            f"{path.name}#{i}: comb score {sf.score:.2f}; moved bass median {np.median(moves['bass']):.1f} px, "
            f"treble median {np.median(moves['treble']):.1f} px"
        )
        fr.extra["board_edge_kpts"] = A.FrameLabel(t=fr.t, anchors=old).kpts_dict()
        fr.anchors = anchors
        note = (
            "D13: re-derived onto the outer strings (scripts/rederive_gt_strings.py: GT wire lines x string comb); "
            "old edge-convention anchors in board_edge_kpts"
        )
        fr.notes = f"{fr.notes}; {note}" if fr.notes else note
        if crops:
            crops.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(crops / f"{lf.clip_id}_f{i:03d}.jpg"), _crops(canon, old, sf, vis2), [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
    A.save(lf, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--crops", type=Path, default=None)
    a = ap.parse_args()
    for p in a.files:
        rederive(p, a.crops)


if __name__ == "__main__":
    main()
