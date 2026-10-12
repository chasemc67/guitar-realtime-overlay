"""Hand-corrected GT frames for eval set v2 (zoom + snap), written to samples/eval_v2/hand/<clip>.json.

Same method as scripts/hand_anchors_0906.py: a few wires whose index is known are clicked on zoomed, grid-labelled
crops of the canonical frame (`scripts/build_eval_v2.py zoom CLIP FRAME --raw`, or scripts/zoom.py); the other wires
of `range` are 12-TET-predicted, grown outwards one at a time and snapped onto their ridge (`gro.snap.snap_wire`);
weak ridges / `occluded` wires keep the model line (vis 1); D13 anchors = wires x outer-string lines from the string
comb. A proof sheet per frame goes to review/eval-v2/hand/.

    .venv/bin/python scripts/hand_anchors_eval_v2.py            # all frames in CLICKS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_v2 import load_config, read_frames  # noqa: E402
from hand_anchors_0906 import MIN_RIDGE, lsq_fit, proof, wire_anchors  # noqa: E402

from gro import anchors as A  # noqa: E402
from gro.auto.detect import prep_gray  # noqa: E402
from gro.auto.strings import detect_strings  # noqa: E402
from gro.geometry import fit_fretboard  # noqa: E402
from gro.snap import intersect, snap_wire  # noqa: E402

OUT = ROOT / "samples" / "eval_v2" / "hand"
PROOF = ROOT / "review" / "eval-v2" / "hand"

# clip -> frame index -> {"wires": {fret: ((bass x, y), (treble x, y))}, "range": (lo, hi), "occluded": [...], "notes"}
# Canonical-frame pixels (D1: mirrored clips are flipped first).
CLICKS: dict[str, dict[int, dict]] = {
    "yt_GEljpsfLj8o": {
        545: {
            # trapezoids in cells 5,7,9,12,15,17,19,21 (wire 4 = right of the cell-5 inlay); 22 = last wire before
            # the board end; nut out of view (right edge). Wide-angle lens: wires visibly bowed.
            "wires": {
                4: ((1180, 388), (1193, 500)),
                5: ((1086, 388), (1097, 510)),
                21: ((392, 442), (375, 558)),
                22: ((372, 442), (354, 558)),
            },
            "range": (4, 22),
        },
    },
}


def build(canon: np.ndarray, spec: dict) -> tuple[list[A.Anchor], dict]:
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
        return (seg, 2) if sc >= spec.get("min_ridge", MIN_RIDGE) else (pred, 1)

    todo = set(range(lo, hi + 1)) - set(segs)
    while todo:
        model = lsq_fit([a for a in wire_anchors(segs) if a.vis == 2])
        n = min(todo, key=lambda k: min(abs(k - f) for f in segs))
        segs[n] = snap(model, n, 4.0)
        todo.discard(n)
    model = lsq_fit([a for a in wire_anchors(segs) if a.vis == 2])
    segs = {n: snap(model, n, 3.0) if segs[n][1] == 2 else segs[n] for n in range(lo, hi + 1)}
    model = fit_fretboard(wire_anchors(segs), tau=None)
    vis2 = sorted(f for f, (_, v) in segs.items() if v == 2)
    sf = detect_strings(gray, model, vis2)
    if sf is None:
        raise RuntimeError("no string comb")
    anchors = []
    for f, (seg, vis) in sorted(segs.items()):
        wl = (seg[0], (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0]))
        for side, ln in zip(A.SIDES, sf.lines, strict=True):
            p = intersect(wl, ln)
            anchors.append(A.Anchor(f, side, round(float(p[0]), 2), round(float(p[1]), 2), vis))
    m = fit_fretboard([a for a in anchors if a.vis == 2], tau=None)
    return anchors, {"scores": scores, "rmse": m.rmse_px, "comb": sf.score}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*")
    a = ap.parse_args()
    cfg = {c["id"]: c for c in load_config()["clips"]}
    OUT.mkdir(parents=True, exist_ok=True)
    PROOF.mkdir(parents=True, exist_ok=True)
    for cid, frames in CLICKS.items():
        if a.clips and cid not in a.clips:
            continue
        clip = cfg[cid]
        imgs = read_frames(ROOT / clip["video"], list(frames), clip["mirrored"])
        out = []
        for fi, spec in sorted(frames.items()):
            canon = imgs[fi]
            anchors, info = build(canon, spec)
            cv2.imwrite(str(PROOF / f"{cid}_{fi}.jpg"), proof(canon, anchors, fi, info), [cv2.IMWRITE_JPEG_QUALITY, 88])
            weak = [f for f, s in info["scores"].items() if s < spec.get("min_ridge", MIN_RIDGE)]
            print(
                f"{cid}#{fi}: wires {len(anchors) // 2}, rmse {info['rmse']:.2f}px, "
                f"comb {info['comb']:.2f}, weak {weak}"
            )
            out.append(
                A.FrameLabel(
                    t=None,
                    anchors=anchors,
                    source="hand",
                    annotator="agent:claude-eval-v2",
                    status="checked",
                    notes="clicked wires " + ",".join(map(str, sorted(spec["wires"]))) + "; " + spec.get("notes", ""),
                    extra={"frame_index": fi},
                )
            )
        h, w = canon.shape[:2]
        video = {"path": clip["video"], "w": w, "h": h, "mirrored": clip["mirrored"]}
        A.save(A.LabelFile(cid, video, {"type": "electric", "model": clip["guitar"]}, out), OUT / f"{cid}.json")


if __name__ == "__main__":
    main()
