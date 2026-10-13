"""Evaluation-only landmark targets: never consulted by inference or voting."""

from __future__ import annotations

import cv2
import numpy as np


def score_detections(detections, positions, present, threshold):
    from gro.fretnet.landmark import LANDMARKS

    scored = []
    for fret, x, visible, d in zip(LANDMARKS, positions, present, detections, strict=True):
        detected = d["confidence"] >= threshold
        error = abs(d["position"] - x) if visible else None
        scored.append(
            dict(
                fret=fret,
                present=bool(visible),
                detected=bool(detected),
                presence_correct=bool((d["presence"] >= 0.5) == visible),
                error_fret=float(error) if error is not None else None,
                localized=bool(visible and detected and error <= 0.5),
            )
        )
    return scored


def frame_diagnostics(backend, image, gt):
    from gro.fretnet.landmark import LANDMARKS
    from gro.fretnet.numbering import HI, LO, rectify
    from gro.geometry import FitError, apply_h, fit_fretboard
    from gro.infer import Letterbox

    refiner = backend.refiner
    row = dict(refiner.last)
    by = {
        f: [g for g in gt if g.fret == f and g.vis > 0 and 0 <= g.x < image.shape[1] and 0 <= g.y < image.shape[0]]
        for f in LANDMARKS
    }
    visible = np.array([bool(by[f]) for f in LANDMARKS])
    mids = np.array([np.mean([[g.x, g.y] for g in by[f]], axis=0) if by[f] else [0, 0] for f in LANDMARKS])
    if row["fit_ok"]:
        if not backend.source_resolution:
            lb = Letterbox.for_shape(image.shape, backend.imgsz)
            mids[:, 0] = mids[:, 0] * lb.new_w / lb.w + lb.left
            mids[:, 1] = mids[:, 1] * lb.new_h / lb.h + lb.top
        plane = apply_h(np.linalg.inv(refiner.fitted_model.H), mids)
        with np.errstate(invalid="ignore", divide="ignore"):
            positions = -12 * np.log2(1 - plane[:, 0])
        present = visible & np.isfinite(positions) & (positions >= LO) & (positions <= HI)
        row["predicted_strip"] = score_detections(row["landmarks"], positions, present, refiner.threshold)
    else:
        # Include detector/fit failures in recall denominators, never silently omit them.
        detections = [dict(position=0, confidence=0, presence=0) for _ in LANDMARKS]
        row["predicted_strip"] = score_detections(detections, LANDMARKS, visible, refiner.threshold)
    try:
        fit = fit_fretboard(gt, ransac_iters=100)
        detections = refiner.detect(rectify(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), fit, refiner.net.width))
        row["gt_strip"] = score_detections(detections, LANDMARKS, visible, refiner.threshold)
    except (FitError, ValueError, np.linalg.LinAlgError):
        row["gt_strip"] = None
    return row


def aggregate_diagnostics(rows):
    from gro.fretnet.landmark import LANDMARKS

    result = {}
    for strip in ("predicted_strip", "gt_strip"):
        output = {}
        for fret in LANDMARKS:
            records = [d for r in rows for d in (r.get("landmark", {}).get(strip) or []) if d["fret"] == fret]
            n = len(records)
            present = sum(d["present"] for d in records)
            detected = sum(d["detected"] for d in records)
            correct = sum(d["localized"] for d in records)
            errors = [d["error_fret"] for d in records if d["present"] and d["detected"]]
            output[str(fret)] = dict(
                frames=n,
                present=present,
                detected=detected,
                localized=correct,
                presence_accuracy=sum(d["presence_correct"] for d in records) / n if n else None,
                precision=correct / detected if detected else None,
                recall=correct / present if present else None,
                detected_mean_error_fret=float(np.mean(errors)) if errors else None,
            )
        result[strip] = output
    result["fit_failures"] = sum(not r.get("landmark", {}).get("fit_ok", False) for r in rows)
    result["nonzero_corrections"] = sum(bool(r.get("landmark", {}).get("offset", 0)) for r in rows)
    return result
