"""Evaluate 50-slot YOLO pose weights on pseudo-GT test images and samples/anchors*.

Requires ultralytics; optional onnxruntime-gpu for CUDA timing. Export the dataset
with scripts/export_yolo.py first (data.yaml may not exist yet). manifest.csv and
source videos are required to recover original pixel dimensions after export resizing.
Example: python scripts/eval_model.py --weights runs/pose/scale-v1/weights/best.pt --name scale-v1
Timing is batch-1 forward inference only (no preprocessing/NMS), 20 warmups + 200
iterations. ONNX defaults to the weights' sibling .onnx; no export is performed here.
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

from gro.anchors import Anchor  # noqa: E402
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
    # Sparse GT is not exhaustive: assess only wires with a visible GT anchor and both endpoints.
    visible_frets = {g.fret for g in visible} & wires.keys()
    matched = defaultdict(list)
    for p in pred:
        if not wires:
            break
        distances = {f: float(abs((np.array([p.x, p.y]) - b) @ n)) for f, (b, n) in wires.items()}
        nearest = min(distances, key=distances.get)
        if nearest in visible_frets and distances[nearest] < 4.0:
            matched[nearest].append(p.fret == nearest)
    numbering = all(matched[f] and all(matched[f]) for f in visible_frets) if visible_frets else None
    return {
        "detected": detected,
        "visible": len(visible),
        "covered": sum((g.fret, g.side) in pm for g in visible),
        "per_keypoint_px": errors,
        "wire_perp_px": perp,
        "outer_string_px": outer,
        "numbering_ok": numbering,
    }


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
        yield path.stem, image, gt, (w / image.shape[1], h / image.shape[0])


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


def benchmark(weights, onnx, imgsz):
    import torch
    from ultralytics import YOLO

    out = {}
    if weights.suffix == ".pt" and torch.cuda.is_available():
        try:
            net = YOLO(str(weights), task="pose").model.to("cuda:0").eval().half()
            x = torch.zeros((1, 3, imgsz, imgsz), device="cuda:0", dtype=torch.float16)
            with torch.inference_mode():
                out["pytorch_fp16_cuda"] = timed(lambda: net(x), torch.cuda.synchronize)
        except (RuntimeError, ValueError) as exc:
            out["pytorch_fp16_cuda"] = {"unavailable": str(exc)}
    else:
        out["pytorch_fp16_cuda"] = {"unavailable": "requires .pt weights and CUDA"}
    try:
        import onnxruntime as ort

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
    return out


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/scale-v1/data.yaml")
    ap.add_argument("--gt", type=Path, action="append")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--onnx", type=Path)
    a = ap.parse_args()
    if not a.dataset.is_file():
        ap.error("dataset YAML absent; run scripts/export_yolo.py first")
    name = a.name or a.weights.stem
    if Path(name).name != name or name in (".", ".."):
        ap.error("--name must be a single filename component")
    from ultralytics import YOLO

    model = YOLO(str(a.weights), task="pose")
    groups = {"pseudo_gt_test": [], "hand_gt": []}
    for item, image, gt, (sx, sy) in test_frames(a.dataset):
        pred, detected = predict(model, image, a.imgsz, a.device)
        pred = [Anchor(p.fret, p.side, p.x * sx, p.y * sy, conf=p.conf) for p in pred]
        groups["pseudo_gt_test"].append({"item": item, **metrics(gt, pred, detected)})
    skipped = []
    for path, lf, i, fr in load_gt(a.gt or sorted((ROOT / "samples").glob("anchors*"))):
        image = frame_image(lf, fr, path)
        if image is None:
            skipped.append(f"{path}#{i}: image unavailable")
            continue
        pred, detected = predict(model, cv2.flip(image, 1) if lf.mirrored else image, a.imgsz, a.device)
        groups["hand_gt"].append({"item": f"{path.name}#{i}", **metrics(fr.anchors, pred, detected)})
    report = clean_json(
        {
            "weights": str(a.weights),
            "imgsz": a.imgsz,
            "skipped": skipped,
            "summary": {k: summarize(v) for k, v in groups.items()},
            "frames": groups,
            "timing": benchmark(a.weights, a.onnx or a.weights.with_suffix(".onnx"), a.imgsz),
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
        "Coverage counts GT vis=2 slots. Detection uses board confidence >= 0.25.",
        "Numbering requires every assessable visible GT wire to have a prediction within 4 px,",
        "with all geometrically matched predictions carrying its correct fret index.",
        "Sparse/unlabelled wires are excluded; missing detections fail numbering.",
        "Missing metric samples are null. Per-frame and per-slot errors are in the JSON.",
        "",
        "| Set | Frames | D13 mean px | Outer mean px | Keypoint mean px | Coverage | Detection | Numbering |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in report["summary"].items():
        values = [
            s["frames"],
            s["wire_perp_px"]["mean"],
            s["outer_string_px"]["mean"],
            s["keypoint_px"]["mean"],
            s["coverage"],
            s["detection_rate"],
            s["numbering_accuracy"],
        ]
        lines.append(f"| {key} | " + " | ".join("n/a" if v is None else f"{v:.3f}" for v in values) + " |")
    lines += [
        "",
        "Batch-1 forward timing: square imgsz, 20 warmups, 200 iterations; excludes preprocessing/NMS.",
        "PyTorch uses CUDA fp16; ONNX uses the graph's dtype with resident CUDA I/O.",
        "```json",
        json.dumps(report["timing"], indent=2),
        "```",
        "",
        f"Skipped hand frames: {len(skipped)}.",
    ]
    Path(f"{destination}.md").write_text("\n".join(lines) + "\n")
    print(Path(f"{destination}.md"))


if __name__ == "__main__":
    main()
