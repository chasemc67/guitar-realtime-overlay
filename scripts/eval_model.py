"""Evaluate 50-slot YOLO pose weights on pseudo-GT test images and samples/anchors*.

Requires ultralytics; optional onnxruntime-gpu / TensorRT engine for timing. Export the dataset
with scripts/export_yolo.py first (data.yaml may not exist yet). manifest.csv and
source videos are required to recover original pixel dimensions after export resizing.
Example: python scripts/eval_model.py --weights runs/pose/scale-v1/weights/best.pt --name scale-v1
Sets: pseudo-GT test split (exported, anchorable frames), pseudo-GT of Chase's held-out 2026-09-06 (every
--stride-0906-th kept frame; its numbering is relative/unresolved in the labels, see the report), and hand GT split
into held-out (2026-09-06) and in-train-videos (everything else). Each set also gets a per-clip breakdown.
Timing is batch-1 forward inference only (no preprocessing/NMS), 20 warmups + 200 iterations, plus an
end-to-end PyTorch predict() timing on a real frame. ONNX/engine default to the weights' siblings; no export here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_auto import frame_image, load_gt, stats  # noqa: E402

from gro.anchors import (
    Anchor,  # noqa: E402
    mirror_anchors,  # noqa: E402
)
from gro.anchors import load as load_label  # noqa: E402
from gro.auto.manifest import clip_info  # noqa: E402
from gro.snap import fit_line  # noqa: E402


def predict(model, image, imgsz=640, device=None):
    """Select the highest-confidence board; preserve absolute 0b,0t,...,24t slots."""
    result = model.predict(image, imgsz=imgsz, device=device, conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], False
    k = int(result.boxes.conf.argmax().item())
    if result.keypoints is None or result.keypoints.conf is None:
        raise ValueError("weights must be a pose model with keypoint confidence")
    xy = result.keypoints.xy[k].cpu().numpy()
    conf = result.keypoints.conf[k].cpu().numpy()
    if xy.shape != (50, 2):
        raise ValueError(f"expected 50 fretboard keypoints, got {xy.shape}")
    anchors = [
        Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
        for i, ((x, y), c) in enumerate(zip(xy, conf, strict=True))
        if c > 0.5 and np.isfinite([x, y, c]).all()
    ]
    return anchors, True


def metrics(gt, pred, detected):
    """Fixed-slot errors; numbering uses independent nearest GT-wire matching."""
    gt = [g for g in gt if g.vis > 0]
    pm = {(p.fret, p.side): p for p in pred}
    by = defaultdict(dict)
    for g in gt:
        by[g.fret][g.side] = np.array([g.x, g.y])
    wires = {}
    for f, sides in by.items():
        if len(sides) == 2:
            b, t = sides["bass"], sides["treble"]
            length = np.linalg.norm(t - b)
            if length > 1e-6:
                wires[f] = (b, np.array([-(t - b)[1], (t - b)[0]]) / length)
    lines = {}
    for side in ("bass", "treble"):
        pts = np.array([[g.x, g.y] for g in gt if g.side == side])
        if len(pts) >= 2 and np.linalg.norm(np.ptp(pts, axis=0)) > 1e-6:
            lines[side] = fit_line(pts)
    errors, perp, outer = {}, [], []
    for g in gt:
        p = pm.get((g.fret, g.side))
        if p is None:
            continue
        q = np.array([p.x, p.y])
        errors[f"{g.fret}{g.side[0]}"] = float(np.linalg.norm(q - [g.x, g.y]))
        if g.fret in wires:
            b, n = wires[g.fret]
            perp.append(float(abs((q - b) @ n)))
        if g.side in lines:
            c, u = lines[g.side]
            outer.append(float(abs((q - c) @ np.array([-u[1], u[0]]))))
    visible = [g for g in gt if g.vis == 2]
    # Numbering: each predicted keypoint is matched to the GT wire it lies on (perpendicular distance below half the
    # local GT fret spacing); it is correct if it carries that wire's index. Frames with no match are not assessable.
    order = sorted(wires)
    shifts = []
    for p in pred:
        if not wires:
            break
        q = np.array([p.x, p.y])
        d = {f: float(abs((q - b) @ n)) for f, (b, n) in wires.items()}
        f = min(d, key=d.get)
        i = order.index(f)
        gaps = [
            abs(float((wires[g][0] - wires[f][0]) @ wires[f][1])) / abs(g - f)
            for g in order[max(i - 1, 0) : i + 2]
            if g != f
        ]
        if gaps and d[f] < 0.5 * min(gaps):
            shifts.append(p.fret - f)
    numbering = (all(s == 0 for s in shifts) if shifts else None) if detected else False
    gt_set = {g.fret for g in gt}  # labelled (vis > 0): the model is trained to infer occluded wires too
    pred_set = {p.fret for p in pred}
    return {
        "detected": detected,
        "visible": len(visible),
        "covered": sum((g.fret, g.side) in pm for g in visible),
        "per_keypoint_px": errors,
        "wire_perp_px": perp,
        "outer_string_px": outer,
        "numbering_ok": numbering,
        "index_shifts": shifts,
        "gt_frets": sorted(gt_set),
        "pred_frets": sorted(pred_set),
        "index_set_exact": gt_set == pred_set if gt_set else None,
        "index_jaccard": len(gt_set & pred_set) / len(gt_set | pred_set) if gt_set else None,
    }


def index_shift(shifts):
    """Per-frame index shift = the most common (pred index - GT index of the wire it lies on), with its support."""
    if not shifts:
        return None, 0.0
    vals, counts = np.unique(shifts, return_counts=True)
    k = int(np.argmax(counts))
    return int(vals[k]), float(counts[k] / len(shifts))


def nearest_wire_px(gt, pred):
    """Index-agnostic alignment: perpendicular px from each predicted keypoint to the nearest GT wire line (only
    predictions within the GT fret range +-1 spacing). Separates "the grid lines up" from "the numbers are right"."""
    by = defaultdict(dict)
    for g in gt:
        if g.vis > 0:
            by[g.fret][g.side] = np.array([g.x, g.y])
    wires = []
    for _f, d in sorted(by.items()):
        if len(d) == 2:
            b, t = d["bass"], d["treble"]
            n = np.array([-(t - b)[1], (t - b)[0]]) / max(np.linalg.norm(t - b), 1e-9)
            wires.append((b, n, (b + t) / 2))
    if len(wires) < 2:
        return []
    mids = np.array([w[2] for w in wires])
    out = []
    for p in pred:
        q = np.array([p.x, p.y])
        d = [abs(float((q - b) @ n)) for b, n, _ in wires]
        i = int(np.argmin(d))
        gap = float(np.linalg.norm(mids[min(i + 1, len(mids) - 1)] - mids[max(i - 1, 0)])) / (
            2 if 0 < i < len(mids) - 1 else 1
        )
        if d[i] < gap:
            out.append(d[i])
    return out


def render_alignment(image, gt, pred, title, path):
    """GT wires (green, index above the low-E string) vs predicted keypoints/wires (magenta, index below the high-E
    string; conf > 0.5 only) on the canonical frame, cropped to the board and upscaled 2x."""
    pts = np.array([[a.x, a.y] for a in [*(g for g in gt if g.vis > 0), *pred]])
    h, w = image.shape[:2]
    x0, y0 = np.clip(pts.min(0) - 60, 0, [w, h]).astype(int)
    x1, y1 = np.clip(pts.max(0) + 60, 0, [w, h]).astype(int)
    z = 2
    out = cv2.resize(image[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    P = lambda a: (int(round((a.x - x0) * z)), int(round((a.y - y0) * z)))  # noqa: E731

    def draw(anchors, col, below):
        by = defaultdict(dict)
        for a in anchors:
            by[a.fret][a.side] = a
        for side in ("bass", "treble"):
            line = [P(d[side]) for f, d in sorted(by.items()) if side in d]
            for p, q in zip(line, line[1:], strict=False):
                cv2.line(out, p, q, col, 1, cv2.LINE_AA)
        for f, d in sorted(by.items()):
            if len(d) == 2:
                cv2.line(out, P(d["bass"]), P(d["treble"]), col, 1, cv2.LINE_AA)
            for a in d.values():
                cv2.circle(out, P(a), 3, col, -1, cv2.LINE_AA)
            a = d.get("treble" if below else "bass")
            if a is not None:
                q = P(a)
                cv2.putText(
                    out, str(f), (q[0] - 6, q[1] + (18 if below else -8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
                )

    draw([g for g in gt if g.vis > 0], (0, 255, 0), False)
    draw(pred, (255, 0, 255), True)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def test_frames(dataset):
    import yaml

    dataset = dataset.resolve()
    config = yaml.safe_load(dataset.read_text())
    root = Path(config.get("path", dataset.parent))
    if not root.is_absolute():
        root = dataset.parent / root
    images = root / config["test"]
    manifest = {r["name"]: r for r in csv.DictReader((root / "manifest.csv").open()) if r["split"] == "test"}
    sizes = {}
    paths = sorted(images.glob("*.jpg"))
    if not paths:
        raise ValueError(f"no exported test images in {images}")
    for path in paths:
        row = manifest[path.stem]
        video = ROOT / row["video"]
        if video not in sizes:
            cap = cv2.VideoCapture(str(video))
            ok, source = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f"cannot recover original resolution from {video}")
            sizes[video] = source.shape[1], source.shape[0]
        w, h = sizes[video]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot read {path}")
        labels = (root / "labels/test" / f"{path.stem}.txt").read_text().splitlines()
        if len(labels) != 1:
            raise ValueError(f"expected one fretboard label: {path}")
        values = np.array([float(v) for v in labels[0].split()])
        if len(values) != 155 or values[0] != 0:
            raise ValueError(f"expected class 0 and 50 keypoints: {path}")
        gt = [
            Anchor(i // 2, ("bass", "treble")[i % 2], x * w, y * h, int(v))
            for i, (x, y, v) in enumerate(values[5:].reshape(50, 3))
            if v > 0
        ]
        yield path.stem, Path(row["video"]).stem, image, gt, (w / image.shape[1], h / image.shape[0])


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _rate(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def by_clip(rows):
    clips = defaultdict(list)
    for r in rows:
        clips[r["clip"]].append(r)
    return {c: {k: v for k, v in summarize(rs).items() if k != "per_keypoint_px"} for c, rs in sorted(clips.items())}


def canonical(image, lf, anchors, override=None):
    """D1: (model input, GT anchors) in the canonical frame. The mirror flag is `override` (CLI), else the clip's
    manifest entry, else the label file's; GT stored under a different flag is re-mirrored to match."""
    info = clip_info(lf.video.get("path", ""))
    mirrored = override if override is not None else info["mirrored"] if info["manifest"] else lf.mirrored
    if mirrored != lf.mirrored:
        anchors = mirror_anchors(list(anchors), image.shape[1])
    return (cv2.flip(image, 1) if mirrored else image), anchors


def heldout_frames(labels, frames_csv, video_stem, stride, mirrored=None):
    """Pseudo-GT frames of a held-out owner video, read at full resolution and canonicalised (D1)."""
    kept = {
        (r["seg_id"], int(r["frame_index"]))
        for r in csv.DictReader(frames_csv.open())
        if r["video"] == video_stem and r["kept"] == "1"
    }
    jobs = []
    for path in sorted(labels.glob(f"{video_stem}_*.json")):
        lf = load_label(path)
        for fr in lf.frames:
            fi = int(fr.extra.get("frame_index", -1))
            if (path.stem, fi) in kept and fr.present():
                jobs.append((fi, lf, fr))
    jobs.sort(key=lambda j: j[0])
    jobs = jobs[::stride]
    if not jobs:
        return
    cap = cv2.VideoCapture(str(ROOT / jobs[0][1].video["path"]))
    try:
        for fi, lf, fr in jobs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, image = cap.read()
            if not ok:
                continue
            yield f"{video_stem}#{fi}", *canonical(image, lf, fr.anchors, mirrored)
    finally:
        cap.release()


def summarize(rows):
    by_slot = defaultdict(list)
    for r in rows:
        for slot, error in r["per_keypoint_px"].items():
            by_slot[slot].append(error)
    numbered = [r["numbering_ok"] for r in rows if r["numbering_ok"] is not None]
    visible = sum(r["visible"] for r in rows)
    return {
        "frames": len(rows),
        "detection_rate": sum(r["detected"] for r in rows) / len(rows) if rows else None,
        "coverage": sum(r["covered"] for r in rows) / visible if visible else None,
        "numbering_accuracy": sum(numbered) / len(numbered) if numbered else None,
        "numbering_frames": len(numbered),
        "index_set_exact": _rate([r["index_set_exact"] for r in rows]),
        "index_jaccard": _mean([r["index_jaccard"] for r in rows]),
        "keypoint_px": stats([e for r in rows for e in r["per_keypoint_px"].values()]),
        "wire_perp_px": stats([e for r in rows for e in r["wire_perp_px"]]),
        "outer_string_px": stats([e for r in rows for e in r["outer_string_px"]]),
        "per_keypoint_px": {k: stats(v) for k, v in sorted(by_slot.items())},
    }


def timed(call, sync=lambda: None):
    for _ in range(20):
        call()
    sync()
    start = time.perf_counter()
    for _ in range(200):
        call()
        sync()
    seconds = time.perf_counter() - start
    return {"fps": 200 / seconds, "ms": seconds * 1000 / 200, "iterations": 200, "warmup": 20}


def benchmark(weights, onnx, imgsz, engine=None, frame=None):
    """Batch-1 timings; any provider that is absent is reported as unavailable."""
    import torch
    from ultralytics import YOLO

    out = {}
    if weights.suffix == ".pt" and torch.cuda.is_available():
        try:
            net = YOLO(str(weights), task="pose").model.fuse().to("cuda:0").eval().half()
            x = torch.zeros((1, 3, imgsz, imgsz), device="cuda:0", dtype=torch.float16)
            with torch.inference_mode():
                out["pytorch_fp16_cuda"] = timed(lambda: net(x), torch.cuda.synchronize)
        except (RuntimeError, ValueError) as exc:
            out["pytorch_fp16_cuda"] = {"unavailable": str(exc)}
    else:
        out["pytorch_fp16_cuda"] = {"unavailable": "requires .pt weights and CUDA"}
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN from the torch wheels; without it the CUDA provider fails to load
        if not onnx.is_file() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("requires sibling/--onnx model and ONNX Runtime CUDA provider")
        session = ort.InferenceSession(str(onnx), providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA provider failed to initialize")
        inp = session.get_inputs()[0]
        dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        x = np.zeros((1, 3, imgsz, imgsz), dtype=dtype)
        # Bind resident CUDA buffers; synchronize output completion without copying to CPU.
        value = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        binding = session.io_binding()
        binding.bind_ortvalue_input(inp.name, value)
        for output in session.get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        out["onnx_cuda"] = {
            **timed(lambda: session.run_with_iobinding(binding), binding.synchronize_outputs),
            "dtype": str(dtype),
        }
    except Exception as exc:  # Optional runtime/provider may be absent or incompatible.
        out["onnx_cuda"] = {"unavailable": str(exc)}
    try:
        if engine is None or not engine.is_file():
            raise RuntimeError("requires sibling/--engine TensorRT engine")
        from ultralytics.nn.autobackend import AutoBackend

        backend = AutoBackend(str(engine), device=torch.device("cuda:0"), fp16=True)
        x = torch.zeros((1, 3, imgsz, imgsz), device="cuda:0", dtype=torch.float16 if backend.fp16 else torch.float32)
        out["tensorrt_fp16"] = timed(lambda: backend(x), torch.cuda.synchronize)
    except Exception as exc:  # TensorRT is optional.
        out["tensorrt_fp16"] = {"unavailable": str(exc)}
    if frame is not None and weights.suffix == ".pt" and torch.cuda.is_available():
        model = YOLO(str(weights), task="pose")
        call = lambda: model.predict(frame, imgsz=imgsz, device=0, half=True, conf=0.25, verbose=False)  # noqa: E731
        out["pytorch_fp16_end_to_end"] = {**timed(call, torch.cuda.synchronize), "input": list(frame.shape)}
    return out


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def table(summary, label="Set"):
    lines = [
        f"| {label} | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        d13 = s["wire_perp_px"]
        cells = [
            str(s["frames"]),
            "n/a" if d13["mean"] is None else f"{d13['mean']:.2f} / {d13['median']:.2f}",
            *("n/a" if v is None else f"{v:.2f}" for v in (s["outer_string_px"]["mean"], s["keypoint_px"]["mean"])),
            *(
                "n/a" if v is None else f"{v:.3f}"
                for v in (
                    s["coverage"],
                    s["detection_rate"],
                    s["numbering_accuracy"],
                    s["index_set_exact"],
                    s["index_jaccard"],
                )
            ),
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/scale-v1/data.yaml")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels/scale-v1")
    ap.add_argument("--heldout", default="2026-09-06_09-00-01", help="owner video evaluated against its pseudo-GT")
    ap.add_argument("--stride-0906", type=int, default=5, help="every n-th kept held-out pseudo-GT frame")
    ap.add_argument("--gt", type=Path, action="append")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument(
        "--align", type=Path, help="visual-alignment mode: render pred vs GT for every held-out hand frame here"
    )
    ap.add_argument(
        "--mirrored", action=argparse.BooleanOptionalAction, default=None,
        help="D1 override for owner/hand-GT videos (default: data_manifests, else the label file's flag)",
    )
    ap.add_argument("--hand-only", action="store_true", help="skip the pseudo-GT sets (hand GT only)")
    a = ap.parse_args()
    if not a.hand_only and not a.dataset.is_file():
        ap.error("dataset YAML absent; run scripts/export_yolo.py first")
    name = a.name or a.weights.stem
    if Path(name).name != name or name in (".", ".."):
        ap.error("--name must be a single filename component")
    from ultralytics import YOLO

    model = YOLO(str(a.weights), task="pose")
    groups = {
        "pseudo_gt_test": [],
        f"pseudo_gt_{a.heldout[:10]}": [],
        "hand_gt_heldout": [],
        "hand_gt_in_train_videos": [],
    }
    for item, clip, image, gt, (sx, sy) in [] if a.hand_only else test_frames(a.dataset):
        pred, detected = predict(model, image, a.imgsz, a.device)
        pred = [Anchor(p.fret, p.side, p.x * sx, p.y * sy, conf=p.conf) for p in pred]
        groups["pseudo_gt_test"].append({"item": item, "clip": clip, **metrics(gt, pred, detected)})
    frame = None
    if (a.labels / "frames.csv").is_file() and not a.hand_only:
        for item, image, gt in heldout_frames(
            a.labels, a.labels / "frames.csv", a.heldout, a.stride_0906, a.mirrored
        ):
            frame = image if frame is None else frame
            pred, detected = predict(model, image, a.imgsz, a.device)
            clip = f"{a.heldout} t={int(item.split('#')[1]) // 60 // 30 * 30:03d}s+"  # 30 s bins at 60 fps
            groups[f"pseudo_gt_{a.heldout[:10]}"].append({"item": item, "clip": clip, **metrics(gt, pred, detected)})
    skipped = []
    for path, lf, i, fr in load_gt(a.gt or sorted((ROOT / "samples").glob("anchors*"))):
        image = frame_image(lf, fr, path)
        if image is None:
            skipped.append(f"{path}#{i}: image unavailable")
            continue
        canon, gt = canonical(image, lf, fr.anchors, a.mirrored)  # D1: GT and model both in the canonical frame
        pred, detected = predict(model, canon, a.imgsz, a.device)
        key = "hand_gt_heldout" if a.heldout in lf.video.get("path", "") else "hand_gt_in_train_videos"
        row = {"item": f"{path.parent.name}/{path.name}#{i}", "clip": path.stem, **metrics(gt, pred, detected)}
        row["index_shift"], row["index_shift_support"] = index_shift(row["index_shifts"])
        row["nearest_wire_px"] = nearest_wire_px(gt, pred)
        groups[key].append(row)
        if a.align and key == "hand_gt_heldout":
            d13, nw = np.median(row["wire_perp_px"]) if row["wire_perp_px"] else np.nan, row["nearest_wire_px"]
            title = (
                f"{path.stem}  GT green / pred magenta  D13(slot) med {d13:.1f}px  nearest-wire med "
                f"{np.median(nw) if nw else np.nan:.1f}px  index shift {row['index_shift']}"
            )
            render_alignment(canon, gt, pred, title, a.align / f"{path.stem}.jpg")
    groups = {k: v for k, v in groups.items() if v}
    timing = (
        {}
        if a.no_timing
        else benchmark(
            a.weights,
            a.onnx or a.weights.with_suffix(".onnx"),
            a.imgsz,
            a.engine or a.weights.with_suffix(".engine"),
            frame,
        )
    )
    report = clean_json(
        {
            "weights": str(a.weights),
            "imgsz": a.imgsz,
            "skipped": skipped,
            "summary": {k: summarize(v) for k, v in groups.items()},
            "per_clip": {k: by_clip(v) for k, v in groups.items()},
            "frames": groups,
            "timing": timing,
        }
    )
    destination = ROOT / "eval" / f"model_{name}_report"
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{destination}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        f"# Model evaluation: {name}",
        "",
        "Errors use original-resolution pixels and absolute slots (vis > 0, prediction conf > 0.5).",
        "D13 headline: perpendicular distance to the GT wire through its bass/treble anchors.",
        "Outer-string error uses a line fitted through each GT side; incomplete wires have no wire error.",
        "Coverage counts GT vis=2 slots predicted with conf > 0.5. Detection uses board confidence >= 0.25.",
        "Numbering: a frame is correct when every prediction lying on a GT wire (within half the local fret spacing)",
        "carries that wire's index; missed detections count as wrong. Index set: predicted fret set == GT labelled (vis>0) set.",
        "Missing metric samples are n/a. Per-frame and per-slot errors are in the JSON.",
        "",
        *table(report["summary"]),
    ]
    for key, clips in report["per_clip"].items():
        lines += ["", f"## {key} per clip", "", *table(clips, "Clip")]
    lines += [
        "",
        "Batch-1 timing: square imgsz, 20 warmups, 200 iterations; forward-only unless *_end_to_end.",
        "```json",
        json.dumps(report["timing"], indent=2),
        "```",
        "",
        f"Skipped hand frames: {len(skipped)}.",
    ]
    if a.align and groups.get("hand_gt_heldout"):
        lines += [
            "",
            "## Visual alignment, held-out hand GT (per frame)",
            "",
            f"Renders: `{a.align.relative_to(ROOT) if a.align.is_absolute() else a.align}/`.",
            "D13(slot): perpendicular px from each predicted slot to the GT wire with the same index.",
            "Nearest-wire: perpendicular px to the closest GT wire whatever its index (does the grid line up?).",
            "Index shift: most common pred index - GT index (support).",
            "",
            "| Frame | Detected | Pred frets | D13 slot median px | Nearest-wire median px | Index shift (support) |",
            "|---|---|---|---:|---:|---:|",
        ]
        for r in groups["hand_gt_heldout"]:
            f = r["pred_frets"]
            med = lambda v: "n/a" if not v else f"{np.median(v):.1f}"  # noqa: E731
            sh = "n/a" if r["index_shift"] is None else f"{r['index_shift']:+d} ({r['index_shift_support']:.0%})"
            lines.append(
                f"| {r['clip']} | {r['detected']} | {f'{f[0]}-{f[-1]} ({len(f)})' if f else '-'} | "
                f"{med(r['wire_perp_px'])} | {med(r['nearest_wire_px'])} | {sh} |"
            )
    Path(f"{destination}.md").write_text("\n".join(lines) + "\n")
    print(Path(f"{destination}.md"))


if __name__ == "__main__":
    main()
