# Model evaluation: v1_0906_honest

Errors use original-resolution pixels and absolute slots (vis > 0, prediction conf > 0.5).
D13 headline: perpendicular distance to the GT wire through its bass/treble anchors.
Outer-string error uses a line fitted through each GT side; incomplete wires have no wire error.
Coverage counts GT vis=2 slots predicted with conf > 0.5. Detection uses board confidence >= 0.25.
Numbering: a frame is correct when every prediction lying on a GT wire (within half the local fret spacing)
carries that wire's index; missed detections count as wrong. Index set: predicted fret set == GT labelled (vis>0) set.
Missing metric samples are n/a. Per-frame and per-slot errors are in the JSON.

| Set | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_gt_heldout | 17 | 0.86 / 0.64 | 1.44 | 1.82 | 0.947 | 1.000 | 1.000 | 0.000 | 0.949 |
| hand_gt_in_train_videos | 3 | 4.06 / 1.76 | 2.34 | 5.21 | 0.917 | 1.000 | 1.000 | 0.000 | 0.803 |

## hand_gt_heldout per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-06_09-00-01_100s | 1 | 1.21 / 1.24 | 0.86 | 1.62 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_10s | 1 | 1.01 / 0.84 | 0.85 | 1.44 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_115s | 1 | 0.82 / 0.79 | 1.27 | 1.58 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_130s | 1 | 0.46 / 0.44 | 1.41 | 1.51 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_145s | 1 | 0.89 / 0.80 | 1.12 | 1.56 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_160s | 1 | 1.01 / 0.62 | 1.38 | 1.94 | 0.917 | 1.000 | 1.000 | 0.000 | 0.917 |
| 2026-09-06_09-00-01_175s | 1 | 0.52 / 0.38 | 0.97 | 1.21 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_195s | 1 | 0.64 / 0.57 | 0.88 | 1.21 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_20s | 1 | 0.74 / 0.54 | 1.07 | 1.42 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_215s | 1 | 1.28 / 1.11 | 1.07 | 1.92 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_240s | 1 | 0.50 / 0.45 | 0.98 | 1.14 | 0.913 | 1.000 | 1.000 | 0.000 | 0.917 |
| 2026-09-06_09-00-01_35s | 1 | 0.63 / 0.40 | 1.18 | 1.42 | 0.917 | 1.000 | 1.000 | 0.000 | 0.917 |
| 2026-09-06_09-00-01_50s | 1 | 0.55 / 0.31 | 0.94 | 1.18 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_5s | 1 | 0.88 / 0.73 | 1.84 | 2.10 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_60s | 1 | 1.29 / 1.28 | 3.27 | 3.72 | 0.905 | 1.000 | 1.000 | 0.000 | 0.917 |
| 2026-09-06_09-00-01_70s | 1 | 1.04 / 1.10 | 1.94 | 2.31 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_85s | 1 | 1.06 / 0.94 | 3.51 | 3.73 | 0.957 | 1.000 | 1.000 | 0.000 | 0.958 |

## hand_gt_in_train_videos per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-30_15-06-39_10s | 1 | 0.99 / 0.66 | 1.78 | 2.07 | 1.000 | 1.000 | 1.000 | 0.000 | 0.609 |
| 2026-09-03_20-38-02_10s | 1 | 3.31 / 1.12 | 1.87 | 4.43 | 0.920 | 1.000 | 1.000 | 0.000 | 0.920 |
| 2026-09-03_20-38-02_60s | 1 | 6.80 / 7.17 | 3.20 | 8.02 | 0.864 | 1.000 | 1.000 | 0.000 | 0.880 |

Batch-1 timing: square imgsz, 20 warmups, 200 iterations; forward-only unless *_end_to_end.
```json
{}
```

Skipped hand frames: 0.

## Visual alignment, held-out hand GT (per frame)

Renders: `review/eval-0906/`.
D13(slot): perpendicular px from each predicted slot to the GT wire with the same index.
Nearest-wire: perpendicular px to the closest GT wire whatever its index (does the grid line up?).
Index shift: most common pred index - GT index (support).

| Frame | Detected | Pred frets | D13 slot median px | Nearest-wire median px | Index shift (support) |
|---|---|---|---:|---:|---:|
| 2026-09-06_09-00-01_10s | True | 1-23 (23) | 0.8 | 0.8 | +0 (100%) |
| 2026-09-06_09-00-01_60s | True | 1-22 (22) | 1.3 | 1.3 | +0 (100%) |
| 2026-09-06_09-00-01_100s | True | 1-23 (23) | 1.2 | 1.2 | +0 (100%) |
| 2026-09-06_09-00-01_115s | True | 1-23 (23) | 0.8 | 0.8 | +0 (100%) |
| 2026-09-06_09-00-01_130s | True | 1-23 (23) | 0.4 | 0.4 | +0 (100%) |
| 2026-09-06_09-00-01_145s | True | 1-23 (23) | 0.8 | 0.8 | +0 (100%) |
| 2026-09-06_09-00-01_160s | True | 1-22 (22) | 0.6 | 0.6 | +0 (100%) |
| 2026-09-06_09-00-01_175s | True | 1-23 (23) | 0.4 | 0.4 | +0 (100%) |
| 2026-09-06_09-00-01_195s | True | 1-23 (23) | 0.6 | 0.6 | +0 (100%) |
| 2026-09-06_09-00-01_20s | True | 1-23 (23) | 0.5 | 0.5 | +0 (100%) |
| 2026-09-06_09-00-01_215s | True | 1-23 (23) | 1.1 | 1.1 | +0 (100%) |
| 2026-09-06_09-00-01_240s | True | 1-22 (22) | 0.5 | 0.5 | +0 (100%) |
| 2026-09-06_09-00-01_35s | True | 1-22 (22) | 0.4 | 0.4 | +0 (100%) |
| 2026-09-06_09-00-01_50s | True | 1-23 (23) | 0.3 | 0.3 | +0 (100%) |
| 2026-09-06_09-00-01_5s | True | 1-23 (23) | 0.7 | 0.7 | +0 (100%) |
| 2026-09-06_09-00-01_70s | True | 1-23 (23) | 1.1 | 1.1 | +0 (100%) |
| 2026-09-06_09-00-01_85s | True | 1-23 (23) | 0.9 | 0.9 | +0 (100%) |
