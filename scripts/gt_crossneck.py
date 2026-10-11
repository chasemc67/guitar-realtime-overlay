"""Cross-neck GT (D13 convention) for BAD batch-002 frames: anchors = fret wire x outer strings (low E, high E).

    uv run python scripts/gt_crossneck.py [--only ID] [--crops review/gt-crossneck]

Model-based outer strings on hand-read board edges (the outer strings themselves are too dim / blurred in this
footage to read to the pixel; the board edges and bindings are clear):
1. `edges`: two points on the bass-side and two on the treble-side edge of the playing surface (inner edge of the
   binding, or where the wires end), read from zoomed, grid-labelled crops (scripts/zoom.py).
2. Wires: the batch-002 auto wires (fret positions + AI numbering were judged mostly right; numbering re-checked here
   against the inlays / nut) are cut by the edges and re-snapped onto their ridge between them (`snap_wire`,
   median objective, so fingers over part of a wire don't matter). Weak ridge -> vis 1 at the model position.
   `drop` removes wrong wires, `occluded` forces vis 1.
3. Outer strings: inset from the edges along each wire by the physical edge->E-string margin, interpolated along the
   neck: board 43 mm at the nut -> 56 mm at the 24th fret, E-to-E span 35.5 -> 52 mm (`gro.auto.strings.span_mm`).
   Typical guitar-to-guitar variation (+-1.5 mm in the margin) makes this GT good to ~3-4% of the span: enough to
   score cross-neck coverage (the BAD frames were off by 30-90%), not sub-pixel string positions.
Where the outer strings themselves are clearly visible they are read directly instead (`strings`, inset 0).
Frames tried and left out: SGTxeiB_jBM (lens curvature: the board edge's local slope is -0.30 at the body end vs a
-0.37 chord, so straight string lines can't be right), QYoiaH5lId8 / HiSaMX45AMw / 3FoeHtEf0P0 (edges + strings not
readable to a few px in the time box; to do).
`--crops` writes an overview (red = low E, blue = high E, grey = board edges, yellow = wires) and zoomed anchor tiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.detect import prep_gray, wire_support
from gro.auto.strings import span_mm
from gro.geometry import fit_fretboard, fret_u
from gro.snap import intersect, robust_line, snap_wire, wire_extent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "anchors_crossneck"
AUTO = ROOT / "labels" / "auto" / "batch-002-auto-yt"
BOARD_NUT_MM, BOARD_24_MM = 43.0, 56.0

READS: dict[str, dict] = {
    "mP2wtBlcCSc": {
        "notes": "dark board, white binding; old auto grid was a sliver at the low-E binding",
        "edges": [[[211.8, 339.6], [1025.1, 185.2]], [[238.5, 462.5], [1042.6, 255.6]]],  # binding centres
    },
    "I4Vt5Q4aUrw": {
        "notes": "dark board, hand at the body end; old grid missed the treble side. Outer strings read directly: "
        "body end past the last fret (six strings visible, 20 px apart), nut end 8% inside the wire ends",
        "strings": [[[346.7, 321.7], [1236.4, 104.4]], [[346.7, 421.7], [1260.8, 179.2]]],
    },
    "IZJdHyfkrsg": {
        "notes": "maple, nut visible, hand covering; old grid missed B + high E. Edges = maple/background boundary",
        "edges": [[[90.0, 359.7], [863.0, 237.0]], [[93.3, 471.3], [863.0, 323.5]]],
    },
}


def _line(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return p, (q - p) / np.linalg.norm(q - p)


def margin_frac(n) -> float:
    """Edge -> outer string margin as a fraction of the board width at fret position n."""
    w = BOARD_NUT_MM + (BOARD_24_MM - BOARD_NUT_MM) * fret_u(n) / fret_u(24)
    return float(0.5 * (w - span_mm(n)) / w)


def auto_edges(gray, by: dict, skip: set):
    """Board edges where the wires end (unbound boards): per wire the ridge extent (`wire_extent`, window grown until
    both ends are found), robust lines through the bass and treble ends."""
    pb, pt = [], []
    for f, d in sorted(by.items()):
        if f in skip or len(d) < 2:
            continue
        sn, _ = snap_wire(gray, d["bass"], d["treble"], search=3.0, step=0.25)
        for ext in (0.35, 1.0, 2.5, 6.0):
            e = wire_extent(gray, sn[0], sn[1], ext=ext)
            if e is not None:
                a, b = np.asarray(e[0]), np.asarray(e[1])
                if np.linalg.norm(a - d["bass"]) > np.linalg.norm(b - d["bass"]):
                    a, b = b, a
                pb.append(a)
                pt.append(b)
                break
    return robust_line(pb)[0], robust_line(pt)[0]


def build(cid: str, r: dict, crops: Path | None):
    img = cv2.imread(str(ROOT / "data/scraped/frames" / f"{cid}_08.jpg"))
    gray = prep_gray(img)
    old = A.load(AUTO / f"{cid}_08.json").frames[0]
    shift = int(r.get("shift", 0))  # correction to the auto numbering, if the inlays say otherwise
    by: dict[int, dict] = {}
    for a in old.anchors:
        if a.fret > 0:
            by.setdefault(a.fret + shift, {})[a.side] = np.array([a.x, a.y])
    if "strings" in r:  # outer strings read directly: inset 0, the "edges" are the strings
        be, te = _line(*r["strings"][0]), _line(*r["strings"][1])
    elif r.get("edges", "auto") == "auto":
        be, te = auto_edges(gray, by, set(r.get("drop", [])) | set(r.get("occluded", [])))
    else:
        be, te = _line(*r["edges"][0]), _line(*r["edges"][1])
    drop, occ = set(r.get("drop", [])), set(r.get("occluded", []))
    segs, vis = {}, {}
    for f, d in sorted(by.items()):
        if f in drop or len(d) < 2:
            continue
        wl = _line(d["bass"], d["treble"])
        b, t = intersect(wl, be), intersect(wl, te)
        sn, _ = snap_wire(gray, b, t, search=3.0, step=0.25)
        med, sup = wire_support(gray, sn[0], sn[1])
        ok = f not in occ and sup >= 0.5 and med >= 4.0
        segs[f], vis[f] = (sn if ok else np.array([b, t])), (2 if ok else 1)
    # Weak wires: from a fit of the strong ones (board-edge convention), so they stay on the 12-TET grid.
    board = []
    for f, sg in segs.items():
        if vis[f] == 2:
            wl = _line(*sg)
            board += [A.Anchor(f, "bass", *map(float, intersect(wl, be)), 2)]
            board += [A.Anchor(f, "treble", *map(float, intersect(wl, te)), 2)]
    model = fit_fretboard(board, tau=None, n_frets=24)
    out = []
    for f in sorted(segs):
        if vis[f] == 1:
            b, t = model.project_fy(np.array([f, f]), np.array([0.0, 1.0]))
            segs[f] = np.array([b, t])
        wl = _line(*segs[f])
        b, t = intersect(wl, be), intersect(wl, te)
        m = 0.0 if "strings" in r else margin_frac(f)
        pb, pt = b + m * (t - b), t - m * (t - b)
        out += [A.Anchor(f, "bass", float(pb[0]), float(pb[1]), vis[f])]
        out += [A.Anchor(f, "treble", float(pt[0]), float(pt[1]), vis[f])]
    fr = A.FrameLabel(
        t=8.0,
        anchors=out,
        source="manual",
        annotator="agent:claude-gt-crossneck",
        status="draft",
        neck_present=True,
        notes=r.get("notes", "") + "; outer strings = board edges inset by the physical margin (see gt_crossneck.py)",
        extra={"frame_file": f"data/scraped/frames/{cid}_08.jpg", "gt_set": "crossneck"},
    )
    fr.anchorable, fr.anchor_cue = old.anchorable, old.anchor_cue
    video = {
        "path": "",
        "source_image": f"data/scraped/frames/{cid}_08.jpg",
        "w": img.shape[1],
        "h": img.shape[0],
        "fps": None,
        "mirrored": False,
    }
    lf = A.LabelFile(f"{cid}_08", video, {"type": "electric", "frets": 24, "strings": 6}, [fr])
    OUT.mkdir(parents=True, exist_ok=True)
    A.save(lf, OUT / f"{cid}_08.json")
    if crops:
        write_crops(img, out, crops / f"{cid}_08", (be, te))
    return lf


def write_crops(img, anchors, stem: Path, board=None, half: int = 22, zoom: int = 5):
    stem.parent.mkdir(parents=True, exist_ok=True)
    ov = img.copy()
    by = {}
    for a in anchors:
        by.setdefault(a.fret, {})[a.side] = a
    for f, d in sorted(by.items()):
        b, t = d["bass"], d["treble"]
        cv2.line(ov, (int(b.x), int(b.y)), (int(t.x), int(t.y)), (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(ov, str(f), (int(b.x) - 6, int(b.y) - 8), 0, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    for side, col in (("bass", (0, 0, 255)), ("treble", (255, 80, 0))):
        pts = np.int32([[d[side].x, d[side].y] for _, d in sorted(by.items())])
        cv2.polylines(ov, [pts], False, col, 1, cv2.LINE_AA)
    for p, u in board or ():
        cv2.line(ov, tuple(map(int, p - 2000 * u)), tuple(map(int, p + 2000 * u)), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imwrite(str(stem) + "_overview.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tiles = []
    base = img.copy()
    for p, u in board or ():  # board edges, dashed so the edge under the line stays visible
        for k in range(-200, 200, 2):
            q0, q1 = p + (10 * k) * u, p + (10 * k + 5) * u
            cv2.line(base, tuple(map(int, q0)), tuple(map(int, q1)), (180, 180, 180), 1)
    for a in sorted(anchors, key=lambda a: (a.side, a.fret)):
        cx, cy = int(round(a.x)), int(round(a.y))
        crop = cv2.copyMakeBorder(base, half, half, half, half, cv2.BORDER_CONSTANT)[
            cy : cy + 2 * half, cx : cx + 2 * half
        ]
        t = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        c = int((a.x - cx + half + 0.5) * zoom), int((a.y - cy + half + 0.5) * zoom)
        cv2.circle(t, c, 7, (0, 255, 0), 1, cv2.LINE_AA)
        lab = f"{a.fret}{a.side[0]} v{a.vis}"
        cv2.putText(t, lab, (4, 16), 0, 0.5, (0, 0, 0), 3)
        cv2.putText(t, lab, (4, 16), 0, 0.5, (255, 255, 255), 1)
        tiles.append(np.pad(t, ((1, 1), (1, 1), (0, 0))))
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(stem) + "_anchors.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--crops", type=Path, default=None)
    a = ap.parse_args()
    for cid, r in READS.items():
        if a.only and cid != a.only:
            continue
        lf = build(cid, r, a.crops)
        fr = lf.frames[0]
        print(cid, len(fr.anchors), "anchors,", sum(x.vis == 2 for x in fr.anchors) // 2, "visible wires,", fr.notes)


if __name__ == "__main__":
    main()
