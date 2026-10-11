"""Render review images for an annotation batch (PLAN D8: every batch is reviewed by Chase via these images).

    uv run python scripts/review_sheet.py samples/anchors --batch batch-000-hand
    uv run python scripts/review_sheet.py labels/ --batch batch-001 --frames-dir data/frames --cols 4 --rows 3

Frames are found via each label frame's `frame_file`, then `--frames-dir/<clip_id>[_fNNN].jpg`, then the clip video
(`video.path`, at `frame_index` or `t`). Output: review/<batch>/{contact_sheet_NN.jpg, frames/, strips/, summary.csv}.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from gro.review import run_review

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path, help="directory of fretboard-kpts/v1 JSON files")
    ap.add_argument("--batch", required=True, help="batch name -> review/<batch>/")
    ap.add_argument("--out", type=Path, default=ROOT / "review")
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--width", type=int, default=2000, help="contact sheet / strip width in px")
    ap.add_argument("--zoom", type=float, default=3.0, help="neck strip magnification")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--warn-px", type=float, default=3.0, help="highlight captions with max error above this")
    a = ap.parse_args()

    res = run_review(
        a.labels,
        a.out,
        a.batch,
        bases=[Path.cwd(), ROOT],
        frames_dir=a.frames_dir,
        cols=a.cols,
        rows=a.rows,
        width=a.width,
        zoom=a.zoom,
        quality=a.quality,
        warn_px=a.warn_px,
    )
    for s in res["items"]:
        mean, mx = s["mean_px"], s["max_px"]
        err = f"mean {mean:.2f}px max {mx:.2f}px @{s['max_at']}" if not math.isnan(mean) else s["error"]
        print(
            f"{s['item']}: {s['n_anchors']} anchors, {err}, "
            f"rejected {s['n_rejected']}, nut {s['nut_visible']}, "
            f"anchor {s['anchor_cue'] or 'none'}"
        )
    total = sum(p.stat().st_size for p in res["files"])
    print(f"wrote {len(res['files'])} files to {res['out']} ({total / 1e6:.2f} MB)")
    for p in res["files"]:
        if p.name.startswith("contact_sheet"):
            print(f"  {p}")


if __name__ == "__main__":
    main()
