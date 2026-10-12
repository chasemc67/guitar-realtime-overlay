"""Build / freeze eval set v2 (brief optimize, task 2): held-out non-Ibanez clips with pipeline pseudo-GT.

    label_scale.py run --videos 'data/clips_eval/*.mp4' ... --split test --out data/labels/eval-v2
    filter_scale.py --labels data/labels/eval-v2 --review /tmp/evalv2_filter --no-sheets
    build_eval_v2.py sample        # -> samples/eval_v2/<clip>.json (+ review/eval-v2/<clip>.jpg)
    build_eval_v2.py strips        # -> review/eval-v2/strips/<clip>.jpg (numbering check, see `strips`)
    build_eval_v2.py zoom CLIP FRAME [--x0 ..]   # full-res crop of one GT frame for spot checks

Selection: per clip, up to --per-clip frames kept by scripts/filter_scale.py, spread evenly over the kept frames (at
least --min-gap s apart). The per-clip table in configs/eval_v2.json (`checks`) records what was verified by eye:
`index_offset` (added to every GT fret index of the clip, when the pipeline numbering is shifted), `drop` /
`drop_ranges` (frames whose GT is wrong and cannot be fixed by an offset), `exclude` (clip not verifiable), and
`verified` (frames checked wire by wire on zooms). GT wires past the clip's `frets` (board-end edge or pickup labelled
as a wire) are removed. Hand-corrected frames (zoom + snap, scripts/hand_anchors_eval_v2.py) replace the
pseudo-GT of the same frame and carry source "hand".

The frozen GT (canonical / D1 coordinates, absolute fret slots) is committed under samples/eval_v2/ so every model is
compared on exactly the same frames; scripts/eval_v2.py reads only configs/eval_v2.json + samples/eval_v2/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro import anchors as A  # noqa: E402

CONFIG = ROOT / "configs" / "eval_v2.json"
GT_DIR = ROOT / "samples" / "eval_v2"
REVIEW = ROOT / "review" / "eval-v2"


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def read_frames(video: Path, indices: list[int], mirrored: bool) -> dict[int, np.ndarray]:
    """Exact frames by sequential decoding (seeking can land on a keyframe), canonical (D1) orientation."""
    out = {}
    cap = cv2.VideoCapture(str(video))
    idx = 0
    for fi in sorted(indices):
        while idx < fi:
            cap.grab()
            idx += 1
        ok, img = cap.read()
        idx += 1
        if ok:
            out[fi] = cv2.flip(img, 1) if mirrored else img
    cap.release()
    return out


def draw_gt(img: np.ndarray, anchors: list[A.Anchor], scale: float = 1.0, thick: int = 1) -> np.ndarray:
    out = img.copy()
    by = defaultdict(dict)
    for a in anchors:
        if a.vis > 0:
            by[a.fret][a.side] = (int(round(a.x * scale)), int(round(a.y * scale)))
    for f, d in sorted(by.items()):
        col = (0, 255, 0) if f % 12 else (0, 128, 255)
        if len(d) == 2:
            cv2.line(out, d["bass"], d["treble"], col, thick, cv2.LINE_AA)
        p = d.get("bass") or d.get("treble")
        cv2.putText(out, str(f), (p[0] - 5, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * max(1, thick), col, thick)
    return out


def sample(a) -> None:
    cfg = load_config()
    labels = ROOT / cfg["labels"]
    rows = list(csv.DictReader((labels / "frames.csv").open()))
    by_clip = defaultdict(list)
    for r in rows:
        if r["kept"] == "1":
            by_clip[r["video"]].append(r)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    for clip in cfg["clips"]:
        cid = clip["id"]
        if a.only and cid not in a.only:
            continue
        check = cfg.get("checks", {}).get(cid, {})
        if check.get("exclude"):
            (GT_DIR / f"{cid}.json").unlink(missing_ok=True)
            print(f"{cid}: excluded ({check['exclude']})")
            continue
        cand = sorted(by_clip.get(cid, []), key=lambda r: float(r["t"]))
        drop = set(check.get("drop", []))
        ranges = check.get("drop_ranges", [])
        cand = [
            r
            for r in cand
            if int(r["frame_index"]) not in drop and not any(lo <= int(r["frame_index"]) <= hi for lo, hi in ranges)
        ]
        picked = []
        for i in np.unique(np.linspace(0, len(cand) - 1, 4 * a.per_clip).round().astype(int)) if cand else []:
            t = float(cand[i]["t"])
            if all(abs(t - float(r["t"])) >= a.min_gap for r in picked):
                picked.append(cand[i])
            if len(picked) == a.per_clip:
                break
        picked.sort(key=lambda r: float(r["t"]))
        offset = int(check.get("index_offset", 0))  # applied before the frets cut, so a +1 shift drops the edge
        frames, segs = [], {}
        for r in picked:
            lf = segs.get(r["seg_id"]) or segs.setdefault(r["seg_id"], A.load(labels / f"{r['seg_id']}.json"))
            fr = next(f for f in lf.frames if int(f.extra.get("frame_index", -1)) == int(r["frame_index"]))
            anchors = [A.Anchor(p.fret + offset, p.side, p.x, p.y, p.vis) for p in fr.anchors if p.vis > 0]
            anchors = [p for p in anchors if 0 <= p.fret <= (clip.get("frets") or 24)]
            frames.append(
                A.FrameLabel(
                    t=fr.t,
                    anchors=anchors,
                    source="pseudo",
                    annotator="auto-pipeline (eval-v2)",
                    status="checked" if int(r["frame_index"]) in set(check.get("verified", [])) else "draft",
                    extra={
                        "frame_index": int(r["frame_index"]),
                        "seg_id": r["seg_id"],
                        "index_offset": offset,
                        "numbering_resolved": r["numbering_resolved"],
                        "pipeline_cue": r["anchor_cue"],
                    },
                )
            )
        hand = GT_DIR / "hand" / f"{cid}.json"
        if hand.exists():  # hand-corrected frames replace pseudo-GT of the same frame
            hl = A.load(hand)
            hmap = {int(f.extra["frame_index"]): f for f in hl.frames}
            frames = [hmap.pop(int(f.extra["frame_index"]), f) for f in frames] + list(hmap.values())
            frames.sort(key=lambda f: int(f.extra["frame_index"]))
        if not frames:
            print(f"{cid}: no kept frames")
            continue
        lf0 = next(iter(segs.values())) if segs else A.load(hand)
        video = {**lf0.video, "path": clip["video"], "mirrored": clip["mirrored"]}
        out = A.LabelFile(cid, video, {"type": "electric", "model": clip["guitar"], "strings": 6}, frames)
        A.save(out, GT_DIR / f"{cid}.json")
        if not a.no_sheet:
            sheet(out, REVIEW / f"{cid}.jpg")
        print(f"{cid}: {len(frames)} frames (of {len(cand)} kept), offset {offset:+d}")


def sheet(lf: A.LabelFile, path: Path, cols: int = 4, width: int = 480) -> None:
    idx = [int(f.extra["frame_index"]) for f in lf.frames]
    imgs = read_frames(ROOT / lf.video["path"], idx, lf.mirrored)
    tiles = []
    for f in lf.frames:
        fi = int(f.extra["frame_index"])
        if fi not in imgs:
            continue
        img = imgs[fi]
        k = width / img.shape[1]
        small = cv2.resize(img, (width, round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        t = draw_gt(small, f.present(), k)
        cv2.putText(t, f"{fi} {f.source}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 2, 0, 2, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def zoom(a) -> None:
    lf = A.load(GT_DIR / f"{a.clip}.json")
    fr = next(f for f in lf.frames if int(f.extra["frame_index"]) == a.frame)
    img = read_frames(ROOT / lf.video["path"], [a.frame], lf.mirrored)[a.frame]
    pts = np.array([[p.x, p.y] for p in fr.present()])
    x0, y0 = pts.min(0) - 40 if a.x0 is None else (a.x0, a.y0)
    x1, y1 = pts.max(0) + 40 if a.x1 is None else (a.x1, a.y1)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    s = a.scale
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    shifted = [A.Anchor(p.fret, p.side, (p.x - x0) * s, (p.y - y0) * s, p.vis) for p in fr.present()]
    out = draw_gt(crop, shifted, 1.0, 1) if not a.raw else crop
    cv2.imwrite(a.out, out)
    print(a.out, out.shape, f"crop x {x0}-{x1} y {y0}-{y1}")


def strips(a) -> None:
    """Numbering check: every GT frame of a clip rectified through its own GT fit into a straight strip whose columns
    are uniform in fret number (frets -0.5..max+1.5); strips are stacked, so with consistent numbering the inlays,
    nut and board end line up in columns across frames. Ticks + numbers mark GT wire indices (12 in orange)."""
    from gro.geometry import fit_fretboard, fret_u

    cfg = {c["id"]: c for c in load_config()["clips"]}
    for cid in a.clips or [p.stem for p in sorted(GT_DIR.glob("*.json"))]:
        lf = A.load(GT_DIR / f"{cid}.json")
        imgs = read_frames(ROOT / cfg[cid]["video"], [int(f.extra["frame_index"]) for f in lf.frames], lf.mirrored)
        px = a.px
        f_lo, f_hi = -1.5, 25.5
        cols = int((f_hi - f_lo) * px)
        fr_n = np.linspace(f_lo, f_hi, cols)
        rows_out = []
        for f in lf.frames:
            fi = int(f.extra["frame_index"])
            try:
                m = fit_fretboard(f.present(), tau=None)
            except Exception:  # noqa: BLE001
                continue
            ys = np.linspace(-0.25, 1.25, a.rows)
            F, Y = np.meshgrid(fr_n, ys)
            u = np.where(F >= 0, fret_u(np.maximum(F, 0)), F * (fret_u(1.0) - fret_u(0.0)))
            pts = m.project(u.ravel(), Y.ravel()).reshape(a.rows, cols, 2).astype(np.float32)
            strip = cv2.remap(imgs[fi], pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderValue=(40, 0, 40))
            for g in sorted({p.fret for p in f.present()}):
                x = int((g - f_lo) * px)
                col = (0, 128, 255) if g % 12 == 0 else (0, 255, 0)
                cv2.line(strip, (x, 0), (x, 6), col, 1)
                cv2.line(strip, (x, a.rows - 7), (x, a.rows - 1), col, 1)
                cv2.putText(strip, str(g), (x - 6, a.rows - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
            cv2.putText(strip, str(fi), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            rows_out.append(np.pad(strip, ((0, 2), (0, 0), (0, 0))))
        if rows_out:
            out = REVIEW / "strips" / f"{cid}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.vstack(rows_out), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample")
    ps.add_argument("--per-clip", type=int, default=24)
    ps.add_argument("--min-gap", type=float, default=0.8, help="seconds between selected frames")
    ps.add_argument("--only", nargs="*")
    ps.add_argument("--no-sheet", action="store_true")
    pz = sub.add_parser("zoom")
    pz.add_argument("clip")
    pz.add_argument("frame", type=int)
    pz.add_argument("--x0", type=float)
    pz.add_argument("--y0", type=float)
    pz.add_argument("--x1", type=float)
    pz.add_argument("--y1", type=float)
    pz.add_argument("--scale", type=float, default=2.0)
    pz.add_argument("--raw", action="store_true")
    pz.add_argument("--out", default="/tmp/evz.jpg")
    pst = sub.add_parser("strips")
    pst.add_argument("clips", nargs="*")
    pst.add_argument("--px", type=int, default=40, help="pixels per fret")
    pst.add_argument("--rows", type=int, default=60)
    a = ap.parse_args()
    {"sample": sample, "zoom": zoom, "strips": strips}[a.cmd](a)


if __name__ == "__main__":
    main()
