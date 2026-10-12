"""Label at scale (brief train_v1, task 1): run the temporal auto-annotation over every clip, in parallel, resumably.

    nohup nice -n 10 .venv/bin/python scripts/label_scale.py run --workers 4 > /tmp/scale_main.log 2>&1 &
    .venv/bin/python scripts/label_scale.py status

Videos: data/clips/yt_*.mp4 (minus manifest-excluded ids) + Chase's 2026-08-30 / 2026-09-03 (mirrored, D1).
Chase's 2026-09-06 is TEST-ONLY: labelled only with --include-test (pseudo-GT for eval), never exported to train/val.
The reference screen recording is never labelled.

Every video is cut into segments of at most --seg-s seconds (a 45 s YouTube clip is one segment); each segment is one
`fretboard-kpts/v1` file `<out>/<video stem>_<t0>-<t1>s.json` (keyframes every --keyframe-s, KLT propagation to
--fps) plus `<out>/_runs/<segment>.json` (timing, AI usage). Segments whose label file exists are skipped, so the job
can be killed and restarted, or moved to another machine (copy the out dir along).

AI (numbering, span check) only through the cached client: numbering at most --ai-num-max times per segment and at
most every 5 s, the span check every --span-every-s. Before each segment the ledger (.cache/ai/usage.jsonl) cost
since the run started is checked against --budget; past it, the remaining segments run without AI (relative
numbering, flagged `index_unresolved`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_RAW = ["2026-08-30_15-06-39", "2026-09-03_20-38-02"]
TEST_RAW = ["2026-09-06_09-00-01"]
LEDGER = ROOT / ".cache" / "ai" / "usage.jsonl"


def videos(include_test: bool, globs: list[str] | None = None, split: str = "trainval") -> list[tuple[Path, bool, str]]:
    """(path, mirrored, split hint) for every video to label. `globs` (repo-relative) replaces the default set; the
    D1 mirrored flag then comes from data_manifests/."""
    from gro.auto.manifest import clip_info, excluded_ids

    if globs:
        paths = sorted({p for g in globs for p in ROOT.glob(g) if p.suffix.lower() == ".mp4"})
        return [(p, clip_info(p)["mirrored"], split) for p in paths if p.stem.removeprefix("yt_") not in excluded_ids()]

    excl = excluded_ids()
    out = []
    for p in sorted((ROOT / "data" / "clips").glob("yt_*.mp4")):
        if p.stem[3:] in excl:
            continue
        out.append((p, False, "trainval"))
    for s in TRAIN_RAW:
        out.append((ROOT / "data" / "raw" / f"{s}.mp4", True, "trainval"))
    if include_test:
        for s in TEST_RAW:
            out.append((ROOT / "data" / "raw" / f"{s}.mp4", True, "test"))
    return out


def duration(p: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(p))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n / fps


def segments(vids, seg_s: float) -> list[dict]:
    segs = []
    for p, mirrored, split in vids:
        d = duration(p)
        n = max(1, round(d / seg_s))
        edges = [d * i / n for i in range(n + 1)]
        for t0, t1 in zip(edges[:-1], edges[1:], strict=True):
            t1 = t1 - 0.05 if t1 >= d - 1e-6 else t1  # the last frame often doesn't decode
            segs.append(
                {
                    "video": p,
                    "mirrored": mirrored,
                    "split": split,
                    "t0": round(t0, 2),
                    "t1": round(t1, 2),
                    "seg_id": f"{p.stem}_{t0:.0f}-{t1:.0f}s",
                }
            )
    return segs


def ledger_cost(since: str) -> float:
    if not LEDGER.exists():
        return 0.0
    tot = 0.0
    for line in LEDGER.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("time", "") >= since:
            tot += r.get("cost_usd") or 0.0
    return tot


def run_one(a) -> None:
    """Worker: one segment -> label file (written atomically) + run record."""
    import cv2

    cv2.setNumThreads(1)
    from gro import anchors as A
    from gro.auto.ai import AIClient
    from gro.auto.pipeline import AutoConfig
    from gro.auto.temporal import TrackConfig, annotate_clip

    ai = None if a.no_ai else AIClient()
    cfg = AutoConfig(frets=24, use_ai=not a.no_ai)
    tcfg = TrackConfig(
        out_fps=a.fps, keyframe_s=a.keyframe_s, ai_max_numbering=a.ai_num_max, ai_span_every_s=a.span_every_s
    )
    t_start = time.time()
    frames = annotate_clip(str(a.video), a.mirrored, a.t0, a.t1, ai, cfg, tcfg, log=lambda s: print(s, flush=True))
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    try:
        vpath = str(a.video.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        vpath = str(a.video)
    video = {"path": vpath, "w": size[0], "h": size[1], "fps": round(fps, 3), "mirrored": a.mirrored}
    lf = A.LabelFile(a.seg_id, video, {"type": "electric", "frets": 24, "strings": 6}, frames)
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / f".{a.seg_id}.json.tmp"
    A.save(lf, tmp)
    run = {
        "seg_id": a.seg_id,
        "video": vpath,
        "split": a.split,
        "t0": a.t0,
        "t1": a.t1,
        "frames": len(frames),
        "keyframes": sum(bool(f.extra.get("keyframe")) for f in frames),
        "seconds": round(time.time() - t_start, 1),
        "ai": ai.usage.summary() if ai else None,
        "fps": a.fps,
        "keyframe_s": a.keyframe_s,
    }
    (a.out / "_runs").mkdir(exist_ok=True)
    (a.out / "_runs" / f"{a.seg_id}.json").write_text(json.dumps(run, indent=1))
    os.replace(tmp, a.out / f"{a.seg_id}.json")
    print(f"done {a.seg_id}: {len(frames)} frames in {run['seconds']} s; AI {run['ai']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=ROOT / "data" / "labels" / "scale-v1")
    common.add_argument("--fps", type=float, default=5.0, help="labelled frames per second (propagation)")
    common.add_argument("--keyframe-s", type=float, default=1.0)
    common.add_argument("--ai-num-max", type=int, default=4, help="AI numbering asks per segment")
    common.add_argument("--span-every-s", type=float, default=5.0, help="AI span check interval")
    common.add_argument("--no-ai", action="store_true")
    pr = sub.add_parser("run", parents=[common])
    pr.add_argument("--workers", type=int, default=4)
    pr.add_argument("--seg-s", type=float, default=45.0)
    pr.add_argument("--include-test", action="store_true", help="also label the test-only 2026-09-06 video")
    pr.add_argument("--videos", action="append", help="repo-relative glob(s) replacing the default video set")
    pr.add_argument("--split", default="trainval", help="split hint recorded for --videos")
    pr.add_argument("--only", default="", help="comma-separated substrings of segment ids")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--budget", type=float, default=12.0, help="USD of AI for this run (ledger) before going no-AI")
    pr.add_argument("--logs", type=Path, default=Path("/tmp/scale_logs"))
    po = sub.add_parser("one", parents=[common])
    po.add_argument("--video", type=Path, required=True)
    po.add_argument("--mirrored", action="store_true")
    po.add_argument("--t0", type=float, required=True)
    po.add_argument("--t1", type=float, required=True)
    po.add_argument("--seg-id", required=True)
    po.add_argument("--split", default="trainval")
    ps = sub.add_parser("status", parents=[common])
    ps.add_argument("--include-test", action="store_true")
    ps.add_argument("--seg-s", type=float, default=45.0)
    ps.add_argument("--videos", action="append")
    ps.add_argument("--split", default="trainval")
    a = ap.parse_args()

    if a.cmd == "one":
        run_one(a)
        return

    segs = segments(videos(a.include_test, a.videos, a.split), a.seg_s)
    done = {p.stem for p in a.out.glob("*.json")}
    todo = [s for s in segs if s["seg_id"] not in done]
    if a.cmd == "status":
        runs = [json.loads(p.read_text()) for p in (a.out / "_runs").glob("*.json")] if (a.out / "_runs").exists() else []
        secs = sum(r["seconds"] for r in runs)
        vid_s = sum(r["t1"] - r["t0"] for r in runs)
        left_s = sum(s["t1"] - s["t0"] for s in todo)
        cost = sum((r.get("ai") or {}).get("cost_usd", 0.0) for r in runs)
        print(f"segments {len(segs) - len(todo)}/{len(segs)} done; {sum(r['frames'] for r in runs)} frames")
        if vid_s:
            print(f"worker-seconds per video-second {secs / vid_s:.1f}; {left_s:.0f} video-s left")
            print(f"AI ${cost:.2f} so far (${cost / vid_s * (vid_s + left_s):.2f} projected)")
        return

    if a.only:
        keys = a.only.split(",")
        todo = [s for s in todo if any(k in s["seg_id"] for k in keys)]
    if a.limit:
        todo = todo[: a.limit]
    a.logs.mkdir(parents=True, exist_ok=True)
    since = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{len(segs)} segments, {len(todo)} to do, {a.workers} workers, AI budget ${a.budget} since {since}", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def job(s):
        no_ai = a.no_ai or ledger_cost(since) >= a.budget
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "one", "--out", str(a.out), "--video", str(s["video"]),
            "--t0", str(s["t0"]), "--t1", str(s["t1"]), "--seg-id", s["seg_id"], "--split", s["split"],
            "--fps", str(a.fps), "--keyframe-s", str(a.keyframe_s), "--ai-num-max", str(a.ai_num_max),
            "--span-every-s", str(a.span_every_s),
        ]  # fmt: skip
        cmd += ["--mirrored"] if s["mirrored"] else []
        cmd += ["--no-ai"] if no_ai else []
        t0 = time.time()
        with (a.logs / f"{s['seg_id']}.log").open("w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
        return s["seg_id"], rc, time.time() - t0, no_ai

    n_ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, s) for s in todo]
        for fu in as_completed(futs):
            sid, rc, sec, no_ai = fu.result()
            n_ok += rc == 0
            print(
                f"{dt.datetime.now():%H:%M:%S} {sid}: rc={rc} {sec:.0f}s{' (no AI)' if no_ai else ''}; "
                f"{n_ok}/{len(todo)} ok; AI ${ledger_cost(since):.2f}",
                flush=True,
            )
    print(f"finished: {n_ok}/{len(todo)} ok; AI ${ledger_cost(since):.2f}", flush=True)


if __name__ == "__main__":
    main()
