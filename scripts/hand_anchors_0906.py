"""Hand GT for Chase's held-out 2026-09-06 video (15 frames spread over the clip), written to samples/anchors_0906/.

Unlike scripts/hand_anchors.py, the clicks here are read off zoomed, grid-labelled crops of the *canonical* (mirrored,
D1) frame (`scripts/zoom.py --mirror`), so no mirroring is applied to them. For each frame a few wires whose true
index is known are clicked (rough point near the bass/low-E end and the treble/high-E end): fret 1 = the first wire
on the body side of the fret wrap (the nut is under the wrap; the extrapolated nut gap lands under it), fret 24 = the
last wire at the end of the board, plus a mid wire. Then:
  1. a 12-TET fretboard model is fitted to the clicked wires and every wire in the frame's fret range is predicted;
  2. each predicted wire is snapped onto its bright ridge (`gro.snap.snap_wire`) and the model refitted (2 rounds);
     wires listed as occluded (hand/finger) or with a weak ridge keep the model line and get vis 1;
  3. D13: anchors = wire line x outer-string lines from the string comb (`gro.auto.strings.detect_strings`).
If a clicked index were wrong, 12-TET would not fit both ends and the snapped wires would miss the ridges, so the
proof sheet (review/gt-0906/) checks the numbering as well as the positions.

    uv run python scripts/hand_anchors_0906.py [T ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.detect import prep_gray
from gro.auto.strings import detect_strings
from gro.geometry import FretboardModel, apply_h, dlt, fit_fretboard, fret_u, model_from_h, plane_point
from gro.snap import intersect, snap_wire

ROOT = Path(__file__).resolve().parents[1]
VIDEO = "data/raw/2026-09-06_09-00-01.mp4"
STEM = "2026-09-06_09-00-01"
FRAMES = ROOT / "samples" / "frames"
OUT = ROOT / "samples" / "anchors_0906"
PROOF = ROOT / "review" / "gt-0906"
GUITAR = {"type": "electric", "frets": 24, "inlay": "vine", "strings": 6, "model": "Ibanez (sunburst)"}
MIN_RIDGE = 6.0  # snapped ridge score below this -> wire not clearly visible (vis 1, model line)

# Canonical-frame pixels: fret -> (rough point at the bass/low-E end, rough point at the treble/high-E end).
# range = (first, last) wire labelled; occluded = wires under the fretting hand / pick hand.
CLICKS: dict[int, dict] = {
    5: {
        "wires": {1: ((984, 320), (996, 385)), 3: ((880, 337.5), (888.5, 405)), 24: ((349, 414), (354, 491))},
        "range": (1, 24),
        "occluded": [],
    },
    20: {
        "wires": {
            1: ((1006.7, 320), (1010, 390)),
            2: ((951.7, 325), (962.3, 391.7)),
            24: ((383.3, 400), (385.7, 478.3)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    35: {
        "wires": {1: ((987.7, 290.7), (990.3, 354.7)), 2: ((924.3, 298), (931, 356.3)), 24: ((364.7, 399), (368, 478))},
        "range": (1, 24),
        "occluded": [],
    },
    50: {
        "wires": {
            1: ((961, 278.3), (970.3, 333.3)),
            2: ((907.7, 283.3), (917.7, 331.7)),
            24: ((348, 399.3), (353, 471.7)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    70: {
        "wires": {
            1: ((984.7, 307), (993, 362)),
            2: ((927.3, 313.7), (934.7, 365.3)),
            24: ((352, 400.3), (360.3, 478.7)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    85: {
        "wires": {
            1: ((986.7, 310.3), (996.7, 375.3)),
            2: ((929, 317), (942.3, 387)),
            24: ((351.7, 401.7), (358.3, 478.3)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    100: {
        "wires": {
            1: ((1017.7, 323), (1026, 384.7)),
            2: ((964.3, 321.3), (971, 389.7)),
            24: ((389, 380), (390.3, 456.7)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    115: {
        "wires": {
            1: ((998.3, 293.3), (1006.7, 356.7)),
            2: ((943.3, 300), (953.3, 363.3)),
            24: ((373.3, 381), (376.7, 457)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    130: {
        "wires": {
            1: ((981, 288.7), (991, 348.7)),
            2: ((928.3, 293.7), (937.7, 355.3)),
            24: ((361.3, 380.3), (363, 456.3)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    145: {
        "wires": {
            1: ((980, 287), (990, 350.3)),
            2: ((924.3, 293.7), (935.7, 358.7)),
            24: ((354.7, 385), (361.3, 463.3)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    160: {
        "wires": {1: ((989.7, 301.3), (999.7, 364.7)), 2: ((934, 308), (944, 373)), 24: ((361.3, 383.3), (368, 461.7))},
        "range": (1, 24),
        "occluded": [],
    },
    175: {
        "wires": {
            1: ((1004.7, 325.3), (1013, 390.3)),
            2: ((949.7, 330.3), (958, 395.3)),
            24: ((370, 393.7), (373.3, 472)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    195: {
        "wires": {1: ((938.3, 238), (948.3, 296.3)), 2: ((883, 251.3), (894.7, 308)), 24: ((339, 382), (342.3, 457))},
        "range": (1, 24),
        "occluded": [],
    },
    215: {
        "wires": {
            1: ((964.7, 277), (974.7, 338.7)),
            2: ((910.7, 282), (920.7, 348.7)),
            24: ((348, 384.7), (353, 459.7)),
        },
        "range": (1, 24),
        "occluded": [],
    },
    240: {
        "wires": {1: ((983.3, 305.3), (995, 372)), 2: ((929, 312), (940, 380.3)), 24: ((369.7, 387), (373, 463.7))},
        "range": (1, 24),
        "occluded": [],
    },
}


def frame(t: int) -> np.ndarray:
    path = FRAMES / f"{STEM}_{t}s.jpg"
    if not path.exists():
        cap = cv2.VideoCapture(str(ROOT / VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS))))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read t={t}")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imread(str(path))


def wire_anchors(segs: dict[int, tuple[np.ndarray, int]]) -> list[A.Anchor]:
    return [
        A.Anchor(f, side, float(p[0]), float(p[1]), vis)
        for f, (seg, vis) in segs.items()
        for side, p in zip(A.SIDES, seg, strict=True)
    ]


def lsq_fit(anchors: list[A.Anchor]) -> FretboardModel:
    """Plain least-squares 12-TET fit (DLT over a taper grid): the clicked/snapped wires are clean, and RANSAC is
    unreliable with only the 3 clicked wires."""
    dst = np.array([[a.x, a.y] for a in anchors])
    best = None
    for tau in np.linspace(0.0, 0.6, 13):
        src = plane_point(fret_u(np.array([a.fret for a in anchors], float)), [a.y_canon for a in anchors], tau)
        H = dlt(src, dst)
        r = float(np.sqrt(np.mean(np.sum((apply_h(H, src) - dst) ** 2, axis=1))))
        if best is None or r < best[0]:
            best = (r, model_from_h(H, float(tau)))
    best[1].rmse_px = best[0]
    return best[1]


def build(t: int, spec: dict) -> tuple[A.LabelFile, np.ndarray, dict]:
    canon = cv2.flip(frame(t), 1)
    gray = prep_gray(canon)
    segs = {f: (np.array(s, float), 2) for f, s in spec["wires"].items()}
    lo, hi = spec["range"]
    occ = set(spec.get("occluded", []))
    scores = {}

    def snap(model, n, search):
        pred = model.project_fy(np.array([n, n]), np.array([-0.05, 1.05]))
        if n in occ:
            return pred, 1
        seg, sc = snap_wire(gray, pred[0], pred[1], search=search)
        scores[n] = sc
        return (seg, 2) if sc >= MIN_RIDGE else (pred, 1)

    # Grow outwards from the clicked wires one wire at a time (refit after each), so every prediction is a short
    # extrapolation from snapped neighbours; then one global re-snap pass on the full fit.
    todo = set(range(lo, hi + 1)) - set(segs)
    while todo:
        model = lsq_fit([a for a in wire_anchors(segs) if a.vis == 2])
        n = min(todo, key=lambda k: min(abs(k - f) for f in segs))
        segs[n] = snap(model, n, 4.0)
        todo.discard(n)
    model = lsq_fit([a for a in wire_anchors(segs) if a.vis == 2])
    segs = {n: snap(model, n, 3.0) if n in segs and segs[n][1] == 2 else segs[n] for n in range(lo, hi + 1)}
    model = fit_fretboard(wire_anchors(segs), tau=None)
    vis2 = sorted(f for f, (_, v) in segs.items() if v == 2)
    sf = detect_strings(gray, model, vis2)
    if sf is None:
        raise RuntimeError(f"t={t}: no string comb")
    anchors = []
    for f, (seg, vis) in sorted(segs.items()):
        wl = (seg[0], (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0]))
        for side, ln in zip(A.SIDES, sf.lines, strict=True):
            p = intersect(wl, ln)
            anchors.append(A.Anchor(f, side, round(float(p[0]), 2), round(float(p[1]), 2), vis))
    m = fit_fretboard([a for a in anchors if a.vis == 2], tau=None)
    fit = {"H": np.round(m.H, 8).tolist(), "tau": m.tau, "rmse_px": round(m.rmse_px, 3), "index_shift": 0}
    name = f"{STEM}_{t}s"
    fr = A.FrameLabel(
        t=float(t),
        anchors=anchors,
        fit=fit,
        source="manual",
        annotator="agent:claude-honest-eval",
        status="draft",
        notes="; ".join(
            filter(
                None,
                [
                    "nut hidden under fret wrap; fret 1 = first wire on the body side of the wrap, fret 24 at the "
                    "board "
                    "end; clicked wires "
                    + ",".join(map(str, sorted(spec["wires"])))
                    + " (scripts/hand_anchors_0906.py), "
                    "other wires 12-TET-predicted then ridge-snapped; D13 outer-string anchors (string comb)",
                    spec.get("notes"),
                ],
            )
        ),
        extra={"frame_file": f"samples/frames/{name}.jpg", "frame_index": int(round(t * 60))},
    )
    lf = A.LabelFile(
        clip_id=name,
        video={"path": VIDEO, "w": canon.shape[1], "h": canon.shape[0], "fps": 60, "mirrored": True},
        guitar=GUITAR,
        frames=[fr],
    )
    return lf, canon, {"scores": scores, "rmse": m.rmse_px, "comb": sf.score, "model": m}


def proof(canon: np.ndarray, anchors: list[A.Anchor], t: int, info: dict) -> np.ndarray:
    """Neck strip (2x) with every GT wire drawn and numbered, plus 6x tiles at fret 1 (wrap side) and the last fret."""
    by = {}
    for a in anchors:
        by.setdefault(a.fret, {})[a.side] = a
    pts = np.array([[a.x, a.y] for a in anchors])
    x0, y0 = np.maximum(pts.min(0) - 40, 0).astype(int)
    x1, y1 = np.minimum(pts.max(0) + 40, [canon.shape[1], canon.shape[0]]).astype(int)
    z = 2
    strip = cv2.resize(canon[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    for f, d in sorted(by.items()):
        b, tr = d["bass"], d["treble"]
        col = (0, 255, 0) if b.vis == 2 else (0, 165, 255)
        p, q = np.array([b.x - x0, b.y - y0]) * z, np.array([tr.x - x0, tr.y - y0]) * z
        u = (q - p) / np.linalg.norm(q - p)
        cv2.line(strip, tuple(np.int32(p - 14 * u)), tuple(np.int32(p - 4 * u)), col, 1, cv2.LINE_AA)
        cv2.line(strip, tuple(np.int32(q + 4 * u)), tuple(np.int32(q + 14 * u)), col, 1, cv2.LINE_AA)
        cv2.circle(strip, tuple(np.int32(p)), 2, (255, 0, 255), -1)
        cv2.circle(strip, tuple(np.int32(q)), 2, (255, 0, 255), -1)
        cv2.putText(strip, str(f), tuple(np.int32(p - 24 * u + [-6, 0])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(
        strip,
        f"t={t}s rmse={info['rmse']:.2f}px comb={info['comb']:.2f}",
        (6, strip.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )
    tiles = []
    frets = sorted(by)
    for f in (frets[0], frets[len(frets) // 2], frets[-1]):
        d = by[f]
        c = np.array([(d["bass"].x + d["treble"].x) / 2, (d["bass"].y + d["treble"].y) / 2]).astype(int)
        h = 45
        pad = cv2.copyMakeBorder(canon, h, h, h, h, cv2.BORDER_CONSTANT)
        tile = cv2.resize(
            pad[c[1] : c[1] + 2 * h, c[0] : c[0] + 2 * h], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
        )
        for g in (f - 1, f, f + 1):
            if g in by:
                e = by[g]
                p = (np.array([e["bass"].x, e["bass"].y]) - c + h) * 4
                q = (np.array([e["treble"].x, e["treble"].y]) - c + h) * 4
                u = (q - p) / np.linalg.norm(q - p)
                col = (0, 255, 0) if g == f else (0, 200, 200)
                cv2.line(tile, tuple(np.int32(p - 30 * u)), tuple(np.int32(p - 8 * u)), col, 1, cv2.LINE_AA)
                cv2.line(tile, tuple(np.int32(q + 8 * u)), tuple(np.int32(q + 30 * u)), col, 1, cv2.LINE_AA)
                cv2.circle(tile, tuple(np.int32(p)), 4, (255, 0, 255), 1)
                cv2.circle(tile, tuple(np.int32(q)), 4, (255, 0, 255), 1)
        cv2.putText(tile, f"fret {f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(np.pad(tile, ((2, 2), (2, 2), (0, 0))))
    row = np.hstack(tiles)
    w = max(row.shape[1], strip.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)))  # noqa: E731
    return np.vstack([pad(strip), pad(row)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("times", nargs="*", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PROOF.mkdir(parents=True, exist_ok=True)
    for t, spec in CLICKS.items():
        if a.times and t not in a.times:
            continue
        lf, canon, info = build(t, spec)
        fr = lf.frames[0]
        A.save(lf, OUT / f"{lf.clip_id}.json")
        cv2.imwrite(str(PROOF / f"{lf.clip_id}.jpg"), proof(canon, fr.anchors, t, info), [cv2.IMWRITE_JPEG_QUALITY, 88])
        weak = [f for f, (s) in info["scores"].items() if s < MIN_RIDGE]
        n2 = sum(a.vis == 2 for a in fr.anchors) // 2
        print(
            f"t={t}: wires {len(fr.anchors) // 2} (vis2 {n2}), rmse {info['rmse']:.2f}px, "
            f"comb {info['comb']:.2f}, weak {weak}"
        )


if __name__ == "__main__":
    main()
