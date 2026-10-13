# Refit validation and shipping gate

Stage one: `v4q_r50_832/last.pt`, 832 input, no learned numbering classifier or wire gates.
Coordinates are preserved. Fit in letterbox coordinates using 100 RANSAC iterations, seed 0;
assign each point to its nearest fitted wire, drop distance >= threshold, keep the highest
confidence on collisions, preserve the side, and omit indices outside 0..24.
Fit failure falls back to the original stage-one points. All other fit knobs stay unchanged.

## VAL-only selection

All 1,498 scale-v1aug VAL images; pipeline GT with numbering from label slots.
No eval-v2 clip ID or 09-06 recording filename appears in this split.
Select by pooled VAL numbering, then coverage, then lower mean D13 wire error.
No held-out score participates in selection; selection.json is written before held-out inference.
Prediction confidence > 0.5; board score >= 0.25. Wire errors are source-image pixels.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.3 (selected) | 1498 | 2.62 / 0.81 | 8.7 | 1.85 | 0.813 | 0.844 | 1.000 |
| threshold 0.35 | 1498 | 2.71 / 0.82 | 9.0 | 1.89 | 0.810 | 0.856 | 1.000 |
| threshold 0.4 | 1498 | 2.80 / 0.83 | 9.4 | 1.92 | 0.797 | 0.865 | 1.000 |
| threshold 0.45 | 1498 | 2.89 / 0.84 | 9.8 | 1.95 | 0.778 | 0.872 | 1.000 |
| threshold 0.5 | 1498 | 2.95 / 0.84 | 10.0 | 1.97 | 0.735 | 0.878 | 1.000 |
| threshold 0.55 | 1498 | 2.95 / 0.84 | 10.0 | 1.97 | 0.735 | 0.878 | 1.000 |
| threshold 0.6 | 1498 | 2.95 / 0.84 | 10.0 | 1.97 | 0.735 | 0.878 | 1.000 |

Selected threshold: **0.30 fret**. VAL fit failures: 0.
Thresholds >=0.50 coincide because nearest-integer distance cannot exceed half a fret.

## One fixed-threshold held-out evaluation

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval v2 macro | 232 | 5.28 / 2.16 | 23.2 | 3.46 | 0.714 | 0.846 | 1.000 |
| eval v2 pooled | 232 | 4.83 / 1.10 | 18.1 | 3.35 | 0.649 | 0.848 | 1.000 |
| 09-06 hand GT | 15 | 0.76 / 0.51 | 3.1 | 0.76 | 1.000 | 0.987 | 1.000 |

Shipping gate: **PASS** (eval-v2 numbering > 0.601 and 09-06 numbering = 1.000).
The earlier evaluation-informed 0.45 result is historical context only; it was not used for selection.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/val_refit.py --device cpu --threads 4
```

CPU float32 inference; predictions are cached before applying any threshold.
See `eval/refit_validation/selection.json` for exact numbers and the frozen protocol;
`refit_val_selected.json` contains every held-out frame's metrics.
Checkpoint SHA-256: `259313d18a20f1d9506ff1b2bd7dfb6f95e8a80c4eb24057af73e427931ea031`.
VAL filename manifest SHA-256: `3ea9279b6dea504527e3294c6a786986593333fe1ddef2bc326c18f3d02f6c9e`.
