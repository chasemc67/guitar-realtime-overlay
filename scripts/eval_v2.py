"""Evaluate pose weights on the frozen eval set v2 (configs/eval_v2.json + samples/eval_v2/) and the 09-06 hand set.

    .venv/bin/python scripts/eval_v2.py --weights runs/pose/v1/weights/best.pt --name v1 [--imgsz 640]
    .venv/bin/python scripts/eval_v2.py --weights runs/fretnet/v3_r50/best.pt --name v3_r50   # no ultralytics

Ultralytics .pt weights go through ultralytics' predict (as before). FretNet checkpoints and .onnx / .mlpackage exports
go through gro.infer (letterbox -> backend -> decode -> unletterbox with clipping), the tracker's own path.

Metrics per frame are scripts/eval_model.py's (canonical D1 frame, absolute slots, prediction conf > 0.5):
D13 = perpendicular px from each predicted slot to the GT wire of the same index (headline), numbering = every
prediction lying on a GT wire carries that wire's index, coverage = GT vis-2 slots predicted, detection = board found.
nearest-wire px is index-agnostic (does the grid line up, whatever the numbers). Errors are in source pixels and also
normalised by the local GT fret spacing (% fret), which makes 480p and 1080p clips comparable.
Aggregates: per clip, `macro` = mean over clips (each clip weighs the same; the headline), `pooled` = over frames.
Output: eval/v2/<name>.json and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_v2 import read_frames  # noqa: E402
from eval_auto import frame_image, load_gt  # noqa: E402
from eval_model import canonical, metrics, nearest_wire_px, predict  # noqa: E402

from gro import anchors as A  # noqa: E402
from gro.fretnet.model import is_checkpoint  # noqa: E402
from gro.infer import Letterbox, decode, load_backend  # noqa: E402
from gro.infer.backends import Backend  # noqa: E402


def predict_backend(backend: Backend, image: np.ndarray) -> tuple[list[A.Anchor], bool]:
    """gro.infer path: same letterbox, best-board decode (score >= 0.25) and clipping as the tracker."""
    lb = Letterbox.for_shape(image.shape, backend.imgsz)
    k, _, _ = decode(backend(cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB)), 0.25)
    if not np.isfinite(k[:, 0]).all():
        return [], False
    k[:, :2] = lb.to_frame(k[:, :2])
    anchors = [
        A.Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
        for i, (x, y, c) in enumerate(k)
        if c > 0.5
    ]
    return anchors, True


def fret_spacing(gt: list[A.Anchor]) -> float | None:
    """Median distance between consecutive GT wire midpoints (px per fret, local scale of the frame)."""
    by = defaultdict(dict)
    for g in gt:
        if g.vis > 0:
            by[g.fret][g.side] = np.array([g.x, g.y])
    mids = {f: (d["bass"] + d["treble"]) / 2 for f, d in by.items() if len(d) == 2}
    fs = sorted(mids)
    gaps = [np.linalg.norm(mids[b] - mids[a]) / (b - a) for a, b in zip(fs, fs[1:], strict=False)]
    return float(np.median(gaps)) if gaps else None


def frame_row(model, image, gt, a, item, clip):
    if a.stage1 is not None:
        k = model.predict_frame(image)
        detected = k is not None and np.isfinite(k).all()
        pred = (
            []
            if not detected
            else [
                A.Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), conf=float(c))
                for i, (x, y, c) in enumerate(k)
                if c > 0.5
            ]
        )
    elif isinstance(model, Backend):
        pred, detected = predict_backend(model, image)
    else:
        pred, detected = predict(model, image, a.imgsz, a.device)
    m = metrics(gt, pred, detected)
    sp = fret_spacing(gt)
    m["fret_px"] = sp
    m["wire_perp_pct"] = [100 * e / sp for e in m["wire_perp_px"]] if sp else []
    m["nearest_wire_px"] = nearest_wire_px(gt, pred)
    row = {"item": item, "clip": clip, **{k: v for k, v in m.items() if k != "per_keypoint_px"}}
    if getattr(a, "numbering", None):
        from gro.fretnet.numbering import rectify
        from gro.geometry import FitError, fit_fretboard

        row["stage2"] = dict(model.refiner.last)
        original = model.last_k.copy()
        original[:, :2] = Letterbox.for_shape(image.shape, model.imgsz).to_frame(original[:, :2])
        from gro.fretnet.numbering import anchors_from_k

        baseline = metrics(gt, anchors_from_k(original), detected)
        shifts = baseline["index_shifts"]
        if shifts:
            values, counts = np.unique(shifts, return_counts=True)
            target = -int(values[counts.argmax()])
            row["stage2"]["target_offset"] = target
            row["stage2"]["target_support"] = float(counts.max() / len(shifts))
            row["stage2"]["uniform_offset_possible"] = len(values) == 1
            row["stage2"]["offset_correct"] = row["stage2"]["offset"] == target
        else:
            row["stage2"]["target_offset"] = None
        row["stage2"]["baseline_numbering"] = baseline["numbering_ok"]
        # GT strip diagnosis is evaluation only; never influences the prediction above.
        try:
            fit = fit_fretboard(gt, ransac_iters=100)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            offset, confidence = model.refiner.classify(rectify(rgb, fit, model.refiner.net.width))
            row["stage2"]["gt_offset"] = offset
            row["stage2"]["gt_confidence"] = confidence
        except (FitError, ValueError, np.linalg.LinAlgError):
            row["stage2"]["gt_offset"] = None
    return row


def agg(rows: list[dict]) -> dict:
    perp = [e for r in rows for e in r["wire_perp_px"]]
    pct = [e for r in rows for e in r["wire_perp_pct"]]
    near = [e for r in rows for e in r["nearest_wire_px"]]
    num = [r["numbering_ok"] for r in rows if r["numbering_ok"] is not None]
    vis = sum(r["visible"] for r in rows)
    f = lambda v, q=None: None if not v else float(np.mean(v) if q is None else np.percentile(v, q))  # noqa: E731
    return {
        "frames": len(rows),
        "d13_mean_px": f(perp),
        "d13_median_px": f(perp, 50),
        "d13_p95_px": f(perp, 95),
        "d13_mean_pct_fret": f(pct),
        "nearest_wire_mean_px": f(near),
        "numbering": sum(num) / len(num) if num else None,
        "numbering_frames": len(num),
        "coverage": sum(r["covered"] for r in rows) / vis if vis else None,
        "detection": sum(r["detected"] for r in rows) / len(rows) if rows else None,
    }


def macro(per_clip: dict[str, dict]) -> dict:
    keys = ["d13_mean_px", "d13_median_px", "d13_mean_pct_fret", "nearest_wire_mean_px", "numbering", "coverage",
            "detection"]  # fmt: skip
    out = {"clips": len(per_clip), "frames": sum(s["frames"] for s in per_clip.values())}
    for k in keys:
        v = [s[k] for s in per_clip.values() if s[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def numbering_diagnostics(rows):
    """Pooled classifier accuracy; dominant GT offset is not necessarily a valid global fix."""
    ds = [r["stage2"] for r in rows if "stage2" in r]
    gt = [d for d in ds if d.get("gt_offset") is not None]
    pred = [d for d in ds if d.get("target_offset") is not None and d["fit_ok"]]

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "frames": len(ds),
        "gt_fit_frames": len(gt),
        "stage1_fit_scored_frames": len(pred),
        "gt_strip_offset_accuracy": mean([d["gt_offset"] == 0 for d in gt]),
        "stage1_strip_offset_accuracy": mean([d["offset_correct"] for d in pred]),
        "zero_offset_baseline_accuracy": mean([d["target_offset"] == 0 for d in pred]),
        "uniform_offset_possible_fraction": mean([d["uniform_offset_possible"] for d in pred]),
        "nonzero_changes": sum(d["applied"] and d["offset"] != 0 for d in ds),
    }


def fmt(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def row_md(name, s):
    return (
        f"| {name} | {s['frames']} | {fmt(s['d13_mean_px'])} / {fmt(s['d13_median_px'])} | "
        f"{fmt(s['d13_mean_pct_fret'], 1)} | {fmt(s['nearest_wire_mean_px'])} | {fmt(s['numbering'], 3)} | "
        f"{fmt(s['coverage'], 3)} | {fmt(s['detection'], 3)} |"
    )


HEADER = [
    "| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "eval_v2.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "v2")
    ap.add_argument("--stage1", type=Path, help="FretNet neck-box checkpoint for RTMPose --weights (no GT boxes)")
    ap.add_argument("--numbering", type=Path, help="Whole-neck stage-2 numbering checkpoint")
    ap.add_argument("--wire-reassign", action="store_true", help="Assign points to robust-fit wires")
    ap.add_argument("--numbering-confidence", type=float, default=0.8)
    a = ap.parse_args()
    if a.stage1 and (a.numbering or a.wire_reassign):
        ap.error("--stage1 (RTMPose) cannot be combined with numbering or wire reassignment")
    if a.stage1 is not None:
        from gro.fretnet.rtmpose import TopDown

        model = TopDown(a.stage1, a.weights, device=a.device or "cuda")
        a.imgsz = model.imgsz
    elif a.numbering:
        model = load_backend(
            a.weights,
            numbering=a.numbering,
            device=a.device,
            threshold=a.numbering_confidence,
            wire_reassign=a.wire_reassign,
        )
        a.imgsz = model.imgsz
    elif a.wire_reassign:
        model = load_backend(a.weights, wire_reassign=True, device=a.device)
        a.imgsz = model.imgsz
    elif a.weights.suffix == ".pt" and not is_checkpoint(a.weights):  # ultralytics weights: ultralytics' predict
        from ultralytics import YOLO

        model = YOLO(str(a.weights), task="pose")
    else:
        model = load_backend(a.weights, **({"device": a.device} if a.device else {}))
        a.imgsz = model.imgsz  # FretNet checkpoints carry their own input size
    cfg = json.loads(a.config.read_text())
    gt_dir = ROOT / cfg["gt_dir"]
    rows, guitar = [], {}
    for clip in cfg["clips"]:
        p = gt_dir / f"{clip['id']}.json"
        if not p.exists() or cfg.get("checks", {}).get(clip["id"], {}).get("exclude"):
            continue
        lf = A.load(p)
        guitar[clip["id"]] = clip["guitar"]
        imgs = read_frames(ROOT / clip["video"], [int(f.extra["frame_index"]) for f in lf.frames], clip["mirrored"])
        for fr in lf.frames:
            fi = int(fr.extra["frame_index"])
            if fi in imgs:
                r = frame_row(model, imgs[fi], fr.present(), a, f"{clip['id']}#{fi}", clip["id"])
                r["source"] = fr.source
                rows.append(r)
    honest = []
    for path, lf, i, fr in load_gt([ROOT / g for g in cfg.get("honest", [])]):
        image = frame_image(lf, fr, path)
        if image is None:
            continue
        canon, gt = canonical(image, lf, fr.anchors)
        honest.append(frame_row(model, canon, [g for g in gt if g.vis > 0], a, f"{path.name}#{i}", path.stem))
    by = defaultdict(list)
    for r in rows:
        by[r["clip"]].append(r)
    per_clip = {c: agg(rs) for c, rs in by.items()}
    hand_rows = [r for r in rows if r["source"] == "hand"]
    report = {
        "weights": str(a.weights),
        "wire_reassign": a.wire_reassign,
        "numbering_weights": str(a.numbering) if a.numbering else None,
        "numbering_confidence": a.numbering_confidence if a.numbering else None,
        "stage1": str(a.stage1) if a.stage1 else None,
        "imgsz": a.imgsz,
        "config": cfg["name"],
        "macro": macro(per_clip),
        "pooled": agg(rows),
        "hand_corrected_frames": agg(hand_rows) if hand_rows else None,
        "honest_0906": agg(honest) if honest else None,
        "stage2_diagnostics": {"eval_v2": numbering_diagnostics(rows), "honest_0906": numbering_diagnostics(honest)}
        if a.numbering
        else None,
        "per_clip": per_clip,
        "guitar": guitar,
        "frames": rows,
        "honest_frames": honest,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [
        f"# Eval v2: {a.name}",
        "",
        f"Weights `{a.weights}`, imgsz {a.imgsz}. Frozen set `{a.config.relative_to(ROOT)}`. Macro = mean over clips.",
        *([f"RTMPose neck boxes predicted by `{a.stage1}`; no ground-truth boxes used."] if a.stage1 else []),
        "",
        *HEADER,
        row_md("**eval v2 macro**", report["macro"]),
        row_md("eval v2 pooled", report["pooled"]),
    ]
    if hand_rows:
        lines.append(row_md("eval v2 hand-corrected frames only", report["hand_corrected_frames"]))
    if honest:
        lines.append(row_md("09-06 hand GT (honest)", report["honest_0906"]))
    if a.numbering:
        lines += [
            "",
            f"Stage-2 checkpoint: `{a.numbering}`, confidence threshold {a.numbering_confidence}.",
            "Classifier diagnostics (pooled; GT strips have target offset zero):",
            "",
            "```json",
            json.dumps(report["stage2_diagnostics"], indent=2),
            "```",
        ]
    lines += ["", "## Per clip", "", *HEADER]
    lines += [row_md(f"{c} ({guitar[c]})", s) for c, s in per_clip.items()]
    (a.out / f"{a.name}.md").write_text("\n".join(lines) + "\n")
    print((a.out / f"{a.name}.md").read_text())


if __name__ == "__main__":
    main()
