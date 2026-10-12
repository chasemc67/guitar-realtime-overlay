"""Fret-wrap check: is the fret index of the auto labels shifted against hand GT on Chase's videos?

For every hand-GT frame of an owner video, take the scale-v1 auto label with the nearest frame index (within
--max-gap frames) and compare it wire by wire: each labelled keypoint is matched to the GT wire it lies on (within half
the local fret spacing, `eval_model.metrics`), the per-frame index shift is the most common (label index - GT index).
A wrap read as the nut gives shift -1 (first visible wire labelled 0); "first visible wire = fret 1" when wire 1 is
hidden would give other offsets. Also reports the index-agnostic nearest-wire px (geometry) and the label's own
anchor cue, the nut keypoint if any, and whether the filter kept the frame.

    python scripts/wrap_check.py [--labels data/labels/scale-v1] [--out eval/wrap_check.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_auto import load_gt  # noqa: E402
from eval_model import index_shift, metrics, nearest_wire_px  # noqa: E402

from gro import anchors as A  # noqa: E402
from gro.geometry import apply_h, fit_fretboard  # noqa: E402

OWNER = ("2026-08-30_15-06-39", "2026-09-03_20-38-02", "2026-09-06_09-00-01")


def label_frames(labels: Path, stem: str):
    out = []
    for p in sorted(labels.glob(f"{stem}_*.json")):
        lf = A.load(p)
        for fr in lf.frames:
            if "frame_index" in fr.extra and fr.present():
                out.append((int(fr.extra["frame_index"]), p.stem, fr))
    return out


def fret_pos(gt: list[A.Anchor], pts: list[A.Anchor]) -> list[float]:
    """Fractional fret number of each point along the neck, from a 12-TET fit to the GT (0 = true nut position,
    extrapolated under the wrap when the nut is hidden)."""
    m = fit_fretboard([g for g in gt if g.vis == 2], tau=None)
    q = apply_h(np.linalg.inv(m.H), np.array([[p.x, p.y] for p in pts]))
    return [float(-12 * np.log2(1 - u)) for u in q[:, 0]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels/scale-v1")
    ap.add_argument("--gt", type=Path, action="append")
    ap.add_argument("--max-gap", type=int, default=6, help="max |label frame - GT frame| (60 fps)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval/wrap_check.json")
    a = ap.parse_args()
    kept = {}
    if (a.labels / "frames.csv").is_file():
        for r in csv.DictReader((a.labels / "frames.csv").open()):
            kept[r["seg_id"], int(r["frame_index"])] = (r["kept"] == "1", r.get("reason", ""))
    cache = {}
    rows = []
    for path, lf, _i, fr in load_gt(a.gt or sorted((ROOT / "samples").glob("anchors*"))):
        stem = Path(lf.video.get("path", "")).stem
        if stem not in OWNER:
            continue
        fps = float(lf.video.get("fps", 60))
        fi = int(fr.extra.get("frame_index", round(float(fr.t) * fps)))
        if stem not in cache:
            cache[stem] = label_frames(a.labels, stem)
        cands = [c for c in cache[stem] if abs(c[0] - fi) <= a.max_gap]
        row = {"gt": f"{path.parent.name}/{path.name}", "video": stem, "gt_frame": fi}
        if not cands:
            rows.append({**row, "label": None})
            continue
        lfi, seg, lab = min(cands, key=lambda c: abs(c[0] - fi))
        m = metrics(fr.anchors, lab.present(), True)
        shift, support = index_shift(m["index_shifts"])
        nw = nearest_wire_px(fr.anchors, lab.present())
        k = kept.get((seg, lfi))
        nut = [p for p in lab.present() if p.fret == 0]
        nut_pos = fret_pos(fr.anchors, nut) if nut and 0 not in m["gt_frets"] else None
        rows.append(
            {
                **row,
                "label": f"{seg}#{lfi}",
                "frame_gap": lfi - fi,
                "index_shift": shift,
                "shift_support": support,
                "shift_counts": dict(Counter(m["index_shifts"])),
                "label_frets": m["pred_frets"],
                "gt_frets": m["gt_frets"],
                "label_has_nut": 0 in m["pred_frets"],
                "label_nut_vis": [p.vis for p in nut],
                "label_nut_fret_pos": nut_pos,
                "label_anchor_cue": lab.anchorability()[1],
                "label_anchorable": lab.anchorability()[0],
                "label_flags": lab.flags,
                "kept": None if k is None else k[0],
                "drop_reason": None if k is None else k[1],
                "nearest_wire_px_median": float(np.median(nw)) if nw else None,
            }
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        "| GT | label | gap | shift (support) | label frets | GT frets | cue | kept | nearest-wire px "
        "| label nut at fret pos |"
    )
    print("|---|---|---:|---:|---|---|---|---|---:|---|")
    for r in rows:
        if r["label"] is None:
            print(f"| {r['gt']} | none within {a.max_gap} frames | | | | | | | |")
            continue
        fr_ = lambda f: f"{f[0]}-{f[-1]} ({len(f)})" if f else "-"  # noqa: E731
        sh = "n/a" if r["index_shift"] is None else f"{r['index_shift']:+d} ({r['shift_support']:.0%})"
        nw = "n/a" if r["nearest_wire_px_median"] is None else f"{r['nearest_wire_px_median']:.1f}"
        print(
            f"| {r['gt']} | {r['label']} | {r['frame_gap']} | {sh} | {fr_(r['label_frets'])} | {fr_(r['gt_frets'])} | "
            f"{r['label_anchor_cue']} | {r['kept']} {r['drop_reason'] or ''} | {nw} | "
            + ("-" if r["label_nut_fret_pos"] is None else ", ".join(f"{v:+.2f}" for v in r["label_nut_fret_pos"]))
            + " |"
        )


if __name__ == "__main__":
    main()
