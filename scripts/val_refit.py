"""VAL-only refit threshold selection, followed by one locked held-out evaluation.

Run from the repo: .venv/bin/python scripts/val_refit.py [--device cpu]
Caches stage-one VAL predictions; never uses held-out scores for selection.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev

from gro.anchors import Anchor
from gro.fretnet.data import read_label
from gro.fretnet.numbering import anchors_from_k, reassign_wires
from gro.geometry import FitError, fit_fretboard
from gro.infer import Letterbox, decode, load_backend
from gro.infer.backends import Backend


def score(gt, k, detected, item):
    pred = anchors_from_k(k)
    m = ev.metrics(gt, pred, detected)
    sp = ev.fret_spacing(gt)
    return dict(
        item=item,
        **{key: val for key, val in m.items() if key != "per_keypoint_px"},
        wire_perp_pct=[100 * e / sp for e in m["wire_perp_px"]] if sp else [],
        nearest_wire_px=ev.nearest_wire_px(gt, pred),
    )


def write_report(out: Path, destination: Path):
    """Render saved results without rerunning inference or changing selection."""
    selection = json.loads((out / "selection.json").read_text())
    held = json.loads((out / "refit_val_selected.json").read_text())
    chosen = selection["chosen_threshold"]
    passed = held["macro"]["numbering"] > 0.601 and held["honest_0906"]["numbering"] == 1.0
    lines = [
        "# Refit validation and shipping gate",
        "",
        "Stage one: `v4q_r50_832/last.pt`, 832 input, no learned numbering classifier or wire gates.",
        "Coordinates are preserved. Fit in letterbox coordinates using 100 RANSAC iterations, seed 0;",
        "assign each point to its nearest fitted wire, drop distance >= threshold, keep the highest",
        "confidence on collisions, preserve the side, and omit indices outside 0..24.",
        "Fit failure falls back to the original stage-one points. All other fit knobs stay unchanged.",
        "",
        "## VAL-only selection",
        "",
        "All 1,498 scale-v1aug VAL images; pipeline GT with numbering from label slots.",
        "No eval-v2 clip ID or 09-06 recording filename appears in this split.",
        "Select by pooled VAL numbering, then coverage, then lower mean D13 wire error.",
        "No held-out score participates in selection; selection.json is written before held-out inference.",
        "Prediction confidence > 0.5; board score >= 0.25. Wire errors are source-image pixels.",
        "",
        *ev.HEADER,
    ]
    for threshold, metrics in selection["val"].items():
        label = f"threshold {threshold}" + (" (selected)" if float(threshold) == chosen else "")
        lines.append(ev.row_md(label, metrics))
    lines += [
        "",
        f"Selected threshold: **{chosen:.2f} fret**. VAL fit failures: {selection['fit_failures']}.",
        "Thresholds >=0.50 coincide because nearest-integer distance cannot exceed half a fret.",
        "",
        "## One fixed-threshold held-out evaluation",
        "",
        *ev.HEADER,
        ev.row_md("eval v2 macro", held["macro"]),
        ev.row_md("eval v2 pooled", held["pooled"]),
        ev.row_md("09-06 hand GT", held["honest_0906"]),
        "",
        f"Shipping gate: **{'PASS' if passed else 'FAIL'}** (eval-v2 numbering > 0.601 and 09-06 numbering = 1.000).",
        "The earlier evaluation-informed 0.45 result is historical context only; it was not used for selection.",
        "",
        "## Reproduction",
        "",
        "```sh",
        "OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/val_refit.py --device cpu --threads 4",
        "```",
        "",
        "CPU float32 inference; predictions are cached before applying any threshold.",
        "See `eval/refit_validation/selection.json` for exact numbers and the frozen protocol;",
        "`refit_val_selected.json` contains every held-out frame's metrics.",
        f"Checkpoint SHA-256: `{selection['weights_sha256']}`.",
        f"VAL filename manifest SHA-256: `{selection['val_manifest_sha256']}`.",
        "",
    ]
    destination.write_text("\n".join(lines))
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/fretnet/v4q_r50_832/last.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--out", type=Path, default=ROOT / "eval/refit_validation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "refit_val_selected.json").exists():
        ap.error("Held-out results already exist; do not rerun or tune against them")
    paths = sorted((a.data / "images/val").glob("*.jpg"))
    assert len(paths) == 1498, f"Expected full VAL split, found {len(paths)}"
    cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
    forbidden = [c["id"] for c in cfg["clips"]] + ["09-06", "09_06"]
    assert not any(s in p.name for p in paths for s in forbidden), "held-out filename in VAL"
    provenance = dict(
        weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        val_manifest_sha256=hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest(),
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        ransac_iters=100,
        seed=0,
        selection="pooled VAL numbering descending, coverage descending, D13 mean ascending",
        gt="scale-v1aug pipeline labels; absolute numbering from label slots",
    )
    (a.out / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    backend = load_backend(a.weights, device=a.device)
    rows = {t: [] for t in provenance["thresholds"]}
    failed = 0
    fixtures = []
    for idx, path in enumerate(paths):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        g = read_label(a.data / "labels/val" / (path.stem + ".txt"), w, h)
        gt = [
            Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), vis=int(v))
            for i, (x, y, v) in enumerate(g)
            if v > 0
        ]
        lb = Letterbox.for_shape(image.shape, backend.imgsz)
        cache = a.out / "cache" / (path.stem + ".npz")
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            z = np.load(cache)
            assert str(z["sha"]) == provenance["weights_sha256"]
            k, detected = z["k"], bool(z["detected"])
        else:
            k, conf, _ = decode(backend(cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB)), 0.25)
            detected = bool(conf >= 0.25 and np.isfinite(k).all())
            np.savez_compressed(cache, k=k, detected=detected, sha=provenance["weights_sha256"])
        fit = None
        if detected:
            try:
                fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
            except (FitError, ValueError, np.linalg.LinAlgError):
                failed += 1
        for t in rows:
            out = reassign_wires(k, fit, tolerance=t, reject_ambiguous=True) if fit is not None else k.copy()
            out[:, :2] = lb.to_frame(out[:, :2])
            rows[t].append(score(gt, out if detected else np.zeros_like(k), detected, path.name))
        if fit is not None and len(fixtures) < 20:
            fixtures.append(dict(item=path.name, k=k.tolist(), H=fit.H.tolist()))
        if idx % 50 == 0:
            print(f"VAL {idx}/{len(paths)}", flush=True)
    summary = {str(t): ev.agg(rs) for t, rs in rows.items()}
    chosen = max(
        rows, key=lambda t: (summary[str(t)]["numbering"], summary[str(t)]["coverage"], -summary[str(t)]["d13_mean_px"])
    )
    selection = dict(**provenance, chosen_threshold=chosen, fit_failures=failed, val=summary)
    (a.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (a.out / "fixtures.json").write_text(json.dumps(fixtures) + "\n")
    print(json.dumps(selection, indent=2), flush=True)

    # The selection is persisted before either held-out set is evaluated.
    class LockedRefit(Backend):
        imgsz = backend.imgsz

        def __call__(self, rgb):
            raw = backend(rgb).copy()
            k, conf, _ = decode(raw)
            if conf >= 0.25 and np.isfinite(k).all():
                try:
                    fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
                    raw[5:, 0] = reassign_wires(k, fit, tolerance=chosen, reject_ambiguous=True).reshape(-1)
                except (FitError, ValueError, np.linalg.LinAlgError):
                    pass
            return raw

    ev.load_backend = lambda *args, **kwargs: LockedRefit()
    sys.argv = ["eval_v2.py", "--weights", str(a.weights), "--name", "refit_val_selected", "--out", str(a.out)]
    ev.main()
    write_report(a.out, a.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
