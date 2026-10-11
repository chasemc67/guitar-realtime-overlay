"""Auto-annotate frames with the classical-CV pipeline (+ targeted AI calls), writing `fretboard-kpts/v1` labels.

Independent stills (one label file per frame):

    uv run python scripts/auto_annotate.py frames --video data/raw/2026-09-06_09-00-01.mp4 --times 20,45,90 \\
        --mirrored --out labels/auto/batch-001
    uv run python scripts/auto_annotate.py frames --images 'data/scraped/frames/*.jpg' --out labels/auto/batch-002

A clip (keyframes + KLT propagation + per-frame re-snap, one label file):

    uv run python scripts/auto_annotate.py clip --video data/raw/2026-09-03_20-38-02.mp4 --t0 30 --t1 33 --mirrored \\
        --out labels/auto/clips

Common options: --no-ai (relative numbering when nothing else resolves it), --offline (cached AI answers only),
--frets 24 (guitar.frets), --no-usability. Frames extracted from videos are saved (original orientation) under
--frames-out (default data/auto/<out dir name>/), which `frame_file` points at, so the review tool finds them:

    uv run python scripts/review_sheet.py labels/auto/batch-001 --batch batch-001-auto-chase
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2

from gro import anchors as A
from gro.auto.ai import AIClient
from gro.auto.pipeline import AutoConfig, annotate_frame
from gro.auto.temporal import TrackConfig, annotate_clip

ROOT = Path(__file__).resolve().parents[1]


def rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def read_at(video: Path, t: float):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (img if ok else None), idx, fps, (w, h)


def video_info(path: Path, mirrored: bool, fps: float, size) -> dict:
    return {"path": rel(path), "w": size[0], "h": size[1], "fps": round(fps, 3), "mirrored": mirrored}


def guitar_info(frets: int | None) -> dict:
    return {"type": "electric", "frets": frets or 24, "strings": 6}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frames", "clip"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--video", type=Path)
        p.add_argument("--mirrored", action="store_true", help="clip-level D1 flag (left-handed / mirrored)")
        p.add_argument("--frets", type=int, default=24)
        p.add_argument("--no-ai", action="store_true")
        p.add_argument("--offline", action="store_true")
        p.add_argument("--model", default=None)
        p.add_argument("--no-usability", action="store_true")
    pf = sub.choices["frames"]
    pf.add_argument("--times", default="", help="comma-separated seconds")
    pf.add_argument("--images", default="", help="glob of still images (e.g. <videoid>_<t>.jpg)")
    pf.add_argument("--limit", type=int, default=0)
    pf.add_argument("--frames-out", type=Path, default=None)
    pc = sub.choices["clip"]
    pc.add_argument("--t0", type=float, required=True)
    pc.add_argument("--t1", type=float, required=True)
    pc.add_argument("--fps", type=float, default=10.0)
    pc.add_argument("--keyframe-s", type=float, default=1.0)
    a = ap.parse_args()

    ai = None if a.no_ai else AIClient(**({"model": a.model} if a.model else {}), offline=a.offline)
    cfg = AutoConfig(frets=a.frets, use_ai=not a.no_ai, ai_usability=not a.no_usability)
    a.out.mkdir(parents=True, exist_ok=True)
    report = []
    t_start = time.time()

    if a.cmd == "frames":
        jobs = []
        if a.video:
            for t in [float(x) for x in a.times.split(",") if x.strip()]:
                jobs.append(("video", a.video, t))
        for p in sorted(glob.glob(a.images)) if a.images else []:
            jobs.append(("image", Path(p), None))
        if a.limit:
            jobs = jobs[: a.limit]
        frames_out = a.frames_out or (ROOT / "data" / "auto" / a.out.name)
        frames_out.mkdir(parents=True, exist_ok=True)
        for kind, src, t in jobs:
            if kind == "video":
                img, idx, fps, size = read_at(src, t)
                if img is None:
                    print(f"skip {src}@{t}: cannot read")
                    continue
                clip_id = f"{src.stem}_{t:g}s"
                ff = frames_out / f"{clip_id}.jpg"
                cv2.imwrite(str(ff), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                video = video_info(src, a.mirrored, fps, size)
                extra = {"frame_file": rel(ff), "frame_index": idx}
            else:
                img = cv2.imread(str(src))
                if img is None:
                    continue
                clip_id = src.stem
                parts = clip_id.rsplit("_", 1)
                t = float(parts[1]) if len(parts) == 2 and parts[1].replace(".", "").isdigit() else None
                video = {"path": "", "source_image": rel(src), "w": img.shape[1], "h": img.shape[0], "fps": None,
                         "mirrored": a.mirrored}
                extra = {"frame_file": rel(src)}
            canon = cv2.flip(img, 1) if a.mirrored else img
            t0 = time.time()
            try:
                res = annotate_frame(canon, ai, cfg, t=t, extra=extra)
            except Exception as e:  # noqa: BLE001 - one bad frame must not stop a batch
                print(f"ERROR {clip_id}: {type(e).__name__}: {e}")
                report.append({"clip_id": clip_id, "error": f"{type(e).__name__}: {e}"})
                continue
            lf = A.LabelFile(clip_id, video, guitar_info(a.frets), [res.label])
            A.save(lf, a.out / f"{clip_id}.json")
            num = res.numbering
            rec = {
                "clip_id": clip_id,
                "anchors": len(res.label.anchors),
                "frets": sorted({x.fret for x in res.label.anchors}),
                "rmse": (res.label.fit or {}).get("rmse_px"),
                "numbering": None if num is None else f"{num.source}/{num.cue}/{num.confidence:.2f}",
                "anchorable": res.label.anchorability(),
                "flags": res.label.flags,
                "usable": None if res.usability is None else res.usability.get("usable"),
                "seconds": round(time.time() - t0, 1),
            }
            report.append(rec)
            print(json.dumps(rec, default=str))
    else:
        cap = cv2.VideoCapture(str(a.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        frames = annotate_clip(str(a.video), a.mirrored, a.t0, a.t1, ai, cfg,
                               TrackConfig(out_fps=a.fps, keyframe_s=a.keyframe_s))
        clip_id = f"{a.video.stem}_{a.t0:g}-{a.t1:g}s"
        lf = A.LabelFile(clip_id, video_info(a.video, a.mirrored, fps, size), guitar_info(a.frets), frames)
        A.save(lf, a.out / f"{clip_id}.json")
        for fr in frames:
            report.append({"t": fr.t, "source": fr.source, "anchors": len(fr.anchors),
                           "rmse": (fr.fit or {}).get("rmse_px"), "flags": fr.flags})
    summary = {
        "cmd": a.cmd,
        "n": len(report),
        "seconds": round(time.time() - t_start, 1),
        "ai": ai.usage.summary() if ai else None,
        "items": report,
    }
    # Next to (not inside) the labels dir: the review tool loads every *.json in it.
    (a.out.parent / f"{a.out.name}_run.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"done: {len(report)} items in {summary['seconds']}s; AI {summary['ai']}")


if __name__ == "__main__":
    main()
