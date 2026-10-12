"""Filter the scale labels (brief train_v1, task 2): per-frame keep/drop + reasons, review sheets, summary.

    .venv/bin/python scripts/filter_scale.py                 # data/labels/scale-v1 -> frames.csv + review/scale-v1/

Drop reasons (every frame gets all that apply; `reason` = the first in this order):
- `no_label`: rejected / no anchors / no fit (detection or tracking failed).
- `ai_unusable`: the AI usability check (asked on a segment's first keyframe) said the segment is unusable
  (not an electric close-up, neck not clear); applies to the whole segment.
- `ai_span_bad`: the AI span check said the grid doesn't cover low E -> high E. The check runs on one keyframe every
  ~5 s; every frame takes the verdict of the nearest checked keyframe of its segment within --span-window s.
- `strings_prior`: the outer strings came from the width-prior fallback (no comb, no in-prior edge inset). Propagated
  frames inherit the strings source of their keyframe.
- `rho_out`: width prior rho (outer-string span vs the local fret spacing, `gro.auto.strings`) outside
  [--rho-lo, --rho-hi] at the first or last labelled wire (any allowed index shift if the numbering is relative).
- `rmse_high`: fit rmse > --rmse px.
- `few_wires`: fewer than --min-wires visible (vis 2) wires.
- `track_bad`: temporal flags `fit_jump`, `wires_lost`, `keyframe_mismatch`.
Kept frames that aren't anchorable (D12, `anchorable: false` / relative numbering) stay in the CSV as kept but are
counted separately: they can't train absolute fret keypoints.

Outputs: `<labels>/frames.csv` (also gzipped into review/scale-v1/frames.csv.gz), review/scale-v1/kept_random.jpg,
dropped_random.jpg (24 random tiles each, captioned clip / time / reason) and summary.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro import anchors as A  # noqa: E402
from gro.auto.strings import _local_scales, allowed_shifts, rho_factor  # noqa: E402
from gro.geometry import FretboardModel  # noqa: E402

REASONS = ["no_label", "ai_unusable", "ai_span_bad", "strings_prior", "rho_out", "rmse_high", "few_wires", "track_bad"]
TRACK_BAD = {"fit_jump", "wires_lost", "keyframe_mismatch"}
TEST_VIDEOS = ("2026-09-06_09-00-01",)


def frame_rho(fr: A.FrameLabel, resolved: bool) -> tuple[float, float] | None:
    """Width prior rho at the first / last wire with both outer-string anchors (best allowed shift)."""
    if not fr.fit:
        return None
    by = defaultdict(dict)
    for a in fr.anchors:
        if a.vis > 0 and a.fret > 0:
            by[a.fret][a.side] = np.array([a.x, a.y])
    frets = sorted(f for f, d in by.items() if len(d) == 2)
    if len(frets) < 2:
        return None
    model = FretboardModel(np.array(fr.fit["H"], float), float(fr.fit.get("tau", 0.3)))
    # Relative numbering: any shift that keeps the wires on the board (same rule as the detector's prior).
    shifts = np.array([0]) if resolved else allowed_shifts(frets, None, 24)
    out = []
    for n in (frets[0], frets[-1]):
        w = float(np.linalg.norm(by[n]["treble"] - by[n]["bass"]))
        su, _ = _local_scales(model, n)
        out.append(w / max(su, 1e-9) * rho_factor(n, shifts))
    rho = np.stack(out, -1)  # S, 2
    k = int(np.argmin(np.abs(rho - 0.95).sum(-1)))
    return float(rho[k, 0]), float(rho[k, 1])


BAD_INDEX = [0]


def analyse(label_dir: Path, a) -> list[dict]:
    rows = []
    for path in sorted(label_dir.glob("*.json")):
        d = json.loads(path.read_text())
        ok = []
        for fr in d["frames"]:
            try:
                for name in fr.get("kpts", {}):
                    A.slot_fret(name)
                ok.append(fr)
            except ValueError:  # fret index beyond 24 = mis-numbered relative run
                BAD_INDEX[0] += 1
        d["frames"] = ok
        lf = A.from_dict(d)
        stem = lf.clip_id
        video_stem = Path(lf.video.get("path", "")).stem
        split = "test" if video_stem in TEST_VIDEOS else "trainval"
        frames = lf.frames
        # Segment-level AI usability (first keyframe).
        unusable = any("ai_unusable" in f.flags for f in frames if f.extra.get("keyframe") or f.source == "pseudo")
        span = [
            (f.t, (f.extra.get("auto") or {}).get("span_check"))
            for f in frames
            if (f.extra.get("auto") or {}).get("span_check") is not None
        ]
        key_src = None
        for f in frames:
            auto = f.extra.get("auto") or {}
            is_key = bool(f.extra.get("keyframe"))
            if is_key:
                key_src = auto.get("strings_source")
            reasons = []
            ok_label = bool(f.anchors) and f.fit is not None and f.status != "rejected"
            if not ok_label:
                reasons.append("no_label")
            if unusable:
                reasons.append("ai_unusable")
            near = [(abs(t - f.t), s) for t, s in span if abs(t - f.t) <= a.span_window]
            span_v = min(near, key=lambda x: x[0])[1] if near else None
            if span_v is not None and span_v.get("covers_all_6") is False:
                reasons.append("ai_span_bad")
            src = auto.get("strings_source") if is_key else key_src
            if ok_label and src == "prior":
                reasons.append("strings_prior")
            resolved = "index_unresolved" not in f.flags
            rho = frame_rho(f, resolved) if ok_label else None
            if ok_label and (rho is None or not (a.rho_lo <= rho[0] <= a.rho_hi and a.rho_lo <= rho[1] <= a.rho_hi)):
                reasons.append("rho_out")
            rmse = (f.fit or {}).get("rmse_px")
            if ok_label and rmse is not None and rmse > a.rmse:
                reasons.append("rmse_high")
            n_w = len({x.fret for x in f.anchors if x.vis == 2 and x.fret > 0})
            if ok_label and n_w < a.min_wires:
                reasons.append("few_wires")
            if set(f.flags) & TRACK_BAD:
                reasons.append("track_bad")
            anchorable, cue = f.anchorability()
            frets = sorted({x.fret for x in f.anchors if x.vis > 0})
            rows.append(
                {
                    "seg_id": stem,
                    "video": video_stem,
                    "split": split,
                    "t": round(f.t, 3) if f.t is not None else "",
                    "frame_index": f.extra.get("frame_index", ""),
                    "keyframe": int(is_key),
                    "source": f.source,
                    "mirrored": int(lf.mirrored),
                    "kept": int(not reasons),
                    "reason": reasons[0] if reasons else "",
                    "all_reasons": ";".join(reasons),
                    "n_wires": n_w,
                    "fret_lo": frets[0] if frets else "",
                    "fret_hi": frets[-1] if frets else "",
                    "rmse": rmse if rmse is not None else "",
                    "rho0": round(rho[0], 3) if rho else "",
                    "rho1": round(rho[1], 3) if rho else "",
                    "anchorable": int(bool(anchorable)),
                    "anchor_cue": cue or "",
                    "numbering_resolved": int(resolved),
                    "span_check": "" if span_v is None else int(bool(span_v.get("covers_all_6"))),
                    "flags": ";".join(f.flags),
                    "_label": str(path),
                    "_idx": frames.index(f),
                }
            )
    return rows


def tile(row: dict, width: int = 480, height: int = 300) -> np.ndarray:
    from gro.review import ReviewItem, board_bbox, caption_bar, load_image, render_annotated
    from gro.geometry import FitError

    lf = A.load(row["_label"])
    fr = lf.frames[row["_idx"]]
    it = ReviewItem(Path(row["_label"]), row["_idx"], lf, fr)
    it.image, _ = load_image(lf, fr, row["_idx"], [ROOT], None)
    cap = [
        f"{row['seg_id']}  t={row['t']}s",
        ("KEPT" if row["kept"] else f"DROP {row['all_reasons']}")
        + f" | wires {row['n_wires']} rmse {row['rmse']} rho {row['rho0']}/{row['rho1']}"
        + f" | {'anchor ' + row['anchor_cue'] if row['anchorable'] else 'relative'}",
    ]
    if it.image is None:
        img = np.full((height, width, 3), 60, np.uint8)
    else:
        if fr.anchors and fr.fit:
            H = np.array(fr.fit["H"], float)
            it.model = FretboardModel(H, float(fr.fit.get("tau", 0.3)), 24)
            try:
                img = render_annotated(it, thickness=1, label_scale=0.0)
                x0, y0, x1, y1 = board_bbox(it, pad=0.08)
                img = img[y0:y1, x0:x1]
            except (FitError, ValueError):
                img = it.image
        else:
            img = it.image
        s = min(width / img.shape[1], height / img.shape[0])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        canvas = np.full((height, width, 3), 30, np.uint8)
        y, x = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        img = canvas
    bg = (32, 32, 32) if row["kept"] else (20, 20, 150)
    return np.vstack([caption_bar(width, cap, 0.42, bg), img])


def sheet(rows: list[dict], out: Path, cols: int = 4) -> None:
    tiles = [tile(r) for r in rows]
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "scale-v1")
    ap.add_argument("--review", type=Path, default=ROOT / "review" / "scale-v1")
    ap.add_argument("--rho-lo", type=float, default=0.75)
    ap.add_argument("--rho-hi", type=float, default=1.25)
    ap.add_argument("--rmse", type=float, default=1.5)
    ap.add_argument("--min-wires", type=int, default=6)
    ap.add_argument("--span-window", type=float, default=3.0)
    ap.add_argument("--n-tiles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()

    rows = analyse(a.labels, a)
    cols = [k for k in rows[0] if not k.startswith("_")]
    with (a.labels / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    a.review.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.review / "frames.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    runs = [json.loads(p.read_text()) for p in sorted((a.labels / "_runs").glob("*.json"))]
    md = summary_md(rows, runs, a)
    (a.review / "summary.md").write_text(md)
    print(md)
    if not a.no_sheets:
        rng = random.Random(a.seed)
        tv = [r for r in rows if r["split"] == "trainval"]
        kept = [r for r in tv if r["kept"]]
        # Dropped sheet: frames that had a label (a "no_label" tile shows nothing useful), else anything dropped.
        dropped = [r for r in tv if not r["kept"] and r["reason"] != "no_label"] or [r for r in tv if not r["kept"]]
        sheet(rng.sample(kept, min(a.n_tiles, len(kept))), a.review / "kept_random.jpg")
        sheet(rng.sample(dropped, min(a.n_tiles, len(dropped))), a.review / "dropped_random.jpg")
        print(f"sheets: {a.review / 'kept_random.jpg'}, {a.review / 'dropped_random.jpg'}")


def summary_md(rows: list[dict], runs: list[dict], a) -> str:
    out = ["# scale-v1: label + filter summary", ""]
    vid_s = sum(r["t1"] - r["t0"] for r in runs)
    secs = sum(r["seconds"] for r in runs)
    cost = sum((r.get("ai") or {}).get("cost_usd", 0.0) for r in runs)
    by_task = Counter()
    for r in runs:
        for k, v in ((r.get("ai") or {}).get("by_task") or {}).items():
            by_task[k + " calls"] += v.get("calls", 0)
            by_task[k + " cached"] += v.get("cached", 0)
    if runs:
        out += [
            f"Settings: keyframes every {runs[0]['keyframe_s']} s, propagation to {runs[0]['fps']} fps. "
            f"{len(runs)} segments, {vid_s:.0f} s of video, {secs / 3600:.1f} worker-hours "
            f"({secs / max(vid_s, 1e-9):.1f} worker-s per video-s). AI ${cost:.2f} "
            f"({', '.join(f'{k} {v}' for k, v in sorted(by_task.items()))}).",
            "",
            f"Filter: rho in [{a.rho_lo}, {a.rho_hi}], rmse <= {a.rmse} px, >= {a.min_wires} visible wires, "
            f"AI span verdict of the nearest checked keyframe within {a.span_window} s.",
            "",
        ]
    for split in ("trainval", "test"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        kept = [r for r in rs if r["kept"]]
        drop = Counter(r["reason"] for r in rs if not r["kept"])
        drop_any = Counter(x for r in rs if not r["kept"] for x in r["all_reasons"].split(";"))
        out += [
            f"## {split}" + (" (2026-09-06, TEST ONLY: pseudo-GT for eval, never trained on)" if split == "test" else ""),
            "",
            "| | count |",
            "|---|---|",
            f"| videos | {len({r['video'] for r in rs})} |",
            f"| segments | {len({r['seg_id'] for r in rs})} |",
            f"| frames labelled | {len(rs)} |",
            f"| keyframes | {sum(r['keyframe'] for r in rs)} |",
            f"| kept | {len(kept)} ({len(kept) / len(rs):.0%}) |",
            f"| kept + anchorable (absolute numbering) | {sum(r['anchorable'] for r in kept)} |",
            f"| kept, NOT anchorable (relative only; excluded from absolute training) | "
            f"{sum(not r['anchorable'] for r in kept)} |",
            f"| videos with >= 1 kept frame | {len({r['video'] for r in kept})} |",
            f"| dropped | {len(rs) - len(kept)} |",
            "",
            "| drop reason | first reason | any reason |",
            "|---|---|---|",
        ]
        out += [f"| {k} | {drop.get(k, 0)} | {drop_any.get(k, 0)} |" for k in REASONS]
        cues = Counter(r["anchor_cue"] or "NONE" for r in kept)
        out += ["", "Anchor cues (kept): " + ", ".join(f"{k} {v}" for k, v in cues.most_common()), ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
