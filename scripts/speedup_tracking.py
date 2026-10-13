"""VAL-selected LK homography; true adjacent-frame pairs for sparse held-out labels."""

# ruff: noqa: E402
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev
from speedup_eval import refit, score

from gro.anchors import Anchor
from gro.infer import Letterbox, decode, load_backend


def propagate(previous, current, k, window, flow_size=0):
    """Track image corners inside the last neck hull, then propagate all retained wires."""
    height, width = previous.shape[:2]
    scale = min(1.0, flow_size / max(height, width)) if flow_size else 1.0
    small_width, small_height = round(width * scale), round(height * scale)
    transform = np.diag([small_width / width, small_height / height, 1.0])
    if scale < 1.0:
        previous = cv2.resize(previous, (small_width, small_height), interpolation=cv2.INTER_AREA)
        image = cv2.resize(current, (small_width, small_height), interpolation=cv2.INTER_AREA)
    else:
        image = current
    gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = (k[:, 2] > 0.5) & np.isfinite(k).all(axis=1)
    points = (k[valid, :2] * [transform[0, 0], transform[1, 1]]).astype(np.float32)
    if len(points) < 4:
        return k.copy(), False
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points).astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 5, mask=mask)
    if corners is None or len(corners) < 4:
        return k.copy(), False
    forward, status, _ = cv2.calcOpticalFlowPyrLK(gray, nxt, corners, None, winSize=(window, window), maxLevel=3)
    backward, backstatus, _ = cv2.calcOpticalFlowPyrLK(nxt, gray, forward, None, winSize=(window, window), maxLevel=3)
    ok = (status.ravel() > 0) & (backstatus.ravel() > 0) & (np.linalg.norm(backward - corners, axis=2).ravel() < 2.0)
    if ok.sum() < 4:
        return k.copy(), False
    cv2.setRNGSeed(0)
    H, inliers = cv2.findHomography(corners[ok], forward[ok], cv2.RANSAC, 3.0)
    if H is None or not np.isfinite(H).all() or inliers.sum() < 4:
        return k.copy(), False
    H = np.linalg.inv(transform) @ H @ transform
    out = k.copy()
    out[:, :2] = cv2.perspectiveTransform(k[:, :2].astype(np.float32)[None], H)[0]
    out[:, 0] = out[:, 0].clip(0, current.shape[1])
    out[:, 1] = out[:, 1].clip(0, current.shape[0])
    return out, True


def predict(backend, image, threshold):
    lb = Letterbox.for_shape(image.shape, backend.imgsz)
    raw = backend(cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB))
    k, conf, _ = decode(refit(raw, threshold))
    k[:, :2] = lb.to_frame(k[:, :2])
    return k, bool(conf >= 0.25 and np.isfinite(k).all())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--resume-selection",
        action="store_true",
        help="Resume a pre-scoring infrastructure failure with frozen VAL selection",
    )
    ap.add_argument("--flow-sizes", default="320,640,0", help="VAL candidate long-edge sizes; 0 = native")
    ap.add_argument("--selection-rule", choices=["error", "latency"], default="latency")
    a = ap.parse_args()
    if a.out.exists() or (a.out.with_suffix(".started").exists() and not a.resume_selection):
        ap.error("Refusing repeated held-out evaluation")
    a.out.with_suffix(".started").write_text("Started\n")
    cv2.setNumThreads(1)
    backend = load_backend(a.model, compute_units="CPU_AND_NE")
    threshold = json.loads(a.selection.read_text())["chosen_threshold"]
    if a.resume_selection:
        selection = json.loads(a.out.with_suffix(".selection.json").read_text())
        chosen = selection["window"]
        flow_size = selection.get("flow_size", 0)
    else:
        # Select window on VAL images with known synthetic 2D motions, before held-out access.
        val = {(window, size): [] for window in (21, 31) for size in map(int, a.flow_sizes.split(","))}
        val_ms = {key: [] for key in val}
        for row in json.loads((a.bundle / "calibration.json").read_text()):
            image = cv2.imread(str(a.bundle / row["image"]))
            k, det = predict(backend, image, threshold)
            valid = k[:, 2] > 0.5
            if not det or not valid.any():
                continue
            for dx, dy in [(2.0, 1.0), (-4.0, 2.0), (8.0, -3.0)]:
                moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]))
                for window, size in val:
                    started = time.perf_counter()
                    out, ok = propagate(image, moved, k, window, size)
                    val_ms[(window, size)].append((time.perf_counter() - started) * 1000)
                    error = (
                        float(np.linalg.norm(out[valid, :2] - k[valid, :2] - [dx, dy], axis=1).mean()) if ok else 1e6
                    )
                    val[(window, size)].append(error)
        eligible = [key for key in val if np.mean(val[key]) <= 1.0]
        if a.selection_rule == "error" or not eligible:
            chosen, flow_size = min(val, key=lambda key: np.mean(val[key]))
        else:
            chosen, flow_size = min(eligible, key=lambda key: np.mean(val_ms[key]))
        selection = dict(
            window=chosen,
            flow_size=flow_size,
            rule=a.selection_rule,
            quality_budget_source_px=1.0,
            val_mean_ms={str(key): float(np.mean(v)) for key, v in val_ms.items()},
            val_mean_px={str(w): float(np.mean(v)) for w, v in val.items()},
            method="VAL synthetic translations; fastest within 1 source pixel mean error, or minimum error",
            fallback="reuse last observations if homography unavailable",
        )
        a.out.with_suffix(".selection.json").write_text(json.dumps(selection, indent=2))
    backend.warmup(20)
    rows = defaultdict(list)
    latencies = []
    flow_times = []
    failed = 0
    excluded = []
    for row in json.loads((a.bundle / "manifest.json").read_text()):
        if not row["previous"]:
            excluded.append(row["item"])
            continue
        previous = cv2.imread(str(a.bundle / row["previous"]))
        current = cv2.imread(str(a.bundle / row["image"]))
        start = time.perf_counter()
        k, det = predict(backend, previous, threshold)
        flow_start = time.perf_counter()
        out, ok = propagate(previous, current, k, chosen, flow_size)
        end = time.perf_counter()
        failed += int(not ok)
        flow_times.append((end - flow_start) * 1000)
        latencies.append((end - start) * 1000 / 2)
        gt = [Anchor(**g) for g in row["gt"]]
        scored = score(gt, out if det else np.zeros_like(out), det, row["item"])
        rows[row["group"]].append(dict(**scored, clip=row["clip"], tracked=ok))
    clips = defaultdict(list)
    for row in rows["eval_v2"]:
        clips[row["clip"]].append(row)
    result = dict(
        selection=selection,
        macro=ev.macro({c: ev.agg(rs) for c, rs in clips.items()}),
        honest_0906=ev.agg(rows["honest_0906"]),
        frames=rows,
        failures=failed,
        excluded_no_predecessor=excluded,
        effective_pair_per_frame_ms=dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies, 95))),
        flow_ms=dict(median=float(np.median(flow_times)), p95=float(np.percentile(flow_times, 95))),
        limitation="Sparse labelled targets each treated as skipped frame, with actual immediate predecessor. "
        "Accuracy is skipped-frame-only; pair cost includes inference/refit/preprocess/flow but excludes video decode. "
        "This is an adjacent-pair proxy, not a continuously labelled alternating video.",
    )
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
