# Speeding up v4q without losing numbering accuracy

No Jam Dashboard app changes, release actions, or git commits. Experiments use the
PC RTX 4090 and the Mac Mini M4. Full protocol: [plan.json](speedup/plan.json).
This report is regenerated as results land; missing candidates are still pending.

Targets: CPU_AND_NE median approximately 8 ms or less, eval_v2 numbering >0.601,
and 09-06 numbering 1.000. Historical v1: 4.25 ms, numbering 0.601 / 1.000;
v4q-832 + 0.30 refit: 28.05 / 28.69 ms, numbering 0.714223 / 1.000,
eval_v2 D13 5.2753 / 2.1632 px, coverage 0.846059;
09-06 D13 0.7621 px, coverage 0.987216. These are prior measurements.

## Measured CoreML results

**Accuracy below comes from actual Mini CoreML CPU_AND_NE outputs**, including
compressed candidates, followed by the frozen refit. Timings show both compute
modes: 20 warmups / 200 calls, batch 1, the existing Python CoreML backend
(including PIL conversion, excluding resize/refit/video decoding). No concurrent
experiment is timed on the Mini. Detection and numbering denominators are in the
per-frame JSON reports; parity uses 32 VAL calibration images, never held-out selection.
Raw CoreML outputs are retained as `.CPU_AND_NE.npz` and `.ALL.npz` beside the reports.


| Candidate | CPU_AND_NE median / p95 ms | ALL median / p95 ms | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|---:|---:|---:|
| r832_int8 | 27.02 / 27.30 | 37.06 / 38.60 | 6.40 / 2.92 | 0.665034 | 0.807718 | 1.705 / 0.933 / 0.947443 |
| resize_416 | 5.49 / 5.58 | 8.32 / 8.56 | 19.66 / 10.04 | 0.116071 | 0.646193 | 9.217 / 0.000 / 0.839489 |
| resize_512 | 8.09 / 8.23 | 10.08 / 10.56 | 9.21 / 5.17 | 0.381345 | 0.771110 | 3.923 / 0.000 / 0.940341 |
| resize_640 | 14.03 / 14.13 | 17.08 / 17.54 | 6.24 / 2.75 | 0.614311 | 0.830186 | 1.465 / 0.867 / 0.977273 |

## Recommendations

No collected CoreML candidate yet meets all three strict targets (≤8.00 ms, >0.601, 1.000).

## PyTorch held-out results

These are separate runtime variants, scored once after VAL selection. Do not treat their accuracy as CoreML accuracy.

| Candidate | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|---:|
| distill_resnet18_640 | 12.239 / 7.865 | 0.511465 | 0.775591 | 2.107 / 0.800 / 0.970170 |
| resize_416 | 17.864 / 10.056 | 0.116071 | 0.643467 | 9.009 / 0.000 / 0.825284 |
| resize_512 | 9.024 / 4.993 | 0.363095 | 0.758734 | 3.889 / 0.000 / 0.954545 |
| resize_640 | 6.151 / 2.656 | 0.598889 | 0.829694 | 1.464 / 0.800 / 0.975852 |

## VAL selection

All refit sweeps use 1,498 VAL images and thresholds 0.30–0.60; choose numbering, then coverage, then lower D13. Every checkpoint is selected on VAL mean keypoint error. Filename guards exclude eval_v2 IDs and 09-06 from train/VAL. 640 was selected for distillation from the resize VAL sweep before held-out scoring. All four students use eight epochs, GT + teacher heatmap/keypoint/query-kernel/confidence losses; fine-tunes use four epochs and GT losses. Compression candidates use VAL numbering plus latency only (see plan).

| Candidate | Selected threshold | VAL numbering | VAL coverage | VAL D13 mean px |
|---|---:|---:|---:|---:|
| resize_416 | 0.30 | 0.357143 | 0.753285 | 8.659 |
| resize_512 | 0.30 | 0.456914 | 0.799323 | 5.600 |
| resize_640 | 0.30 | 0.750334 | 0.838868 | 3.091 |
| distill_resnet18_640 | 0.30 | 0.693591 | 0.879366 | 3.452 |

## Tracking

Each sparse labelled target is paired with its actual preceding decoded video frame. Inference/refit on that predecessor is propagated by LK optical flow and a RANSAC homography. This reports **skipped-frame-only accuracy**, not accuracy of a densely labelled full video. Effective adjacent-pair time is (inference + preprocess + Python refit + tracking) / 2, excluding decode. It includes Python postprocessing overhead and is not a native app FPS claim. Window 21/31 is selected on VAL synthetic translations before held-out frames.

| Candidate | Effective pair median / p95 ms per frame | Flow median / p95 ms | eval_v2 D13 mean / median px | Numbering / coverage | 09-06 D13 / numbering / coverage | Flow failures |
|---|---:|---:|---:|---:|---:|---:|

Contiguous-video replay uses alternating inference/flow on up to 120 frames each from 09-06, nBpe2 and Stan. There is no dense ground truth; model-to-model agreement in the raw reports is **not** an accuracy measurement.

| Candidate / clip | Frames | Actual alternating pair median / p95 ms per frame | Flow failures |
|---|---:|---:|---:|

## ALL compute-mode accuracy

| Candidate | CPU_AND_NE median / p95 ms | ALL median / p95 ms | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|---:|---:|---:|
| r832_int8 | 27.02 / 27.30 | 37.06 / 38.60 | 5.95 / 2.85 | 0.637708 | 0.808190 | 1.715 / 0.933 / 0.953125 |
| resize_416 | 5.49 / 5.58 | 8.32 / 8.56 | 17.97 / 9.45 | 0.255952 | 0.638830 | 8.945 / 0.000 / 0.822443 |
| resize_512 | 8.09 / 8.23 | 10.08 / 10.56 | 9.53 / 5.24 | 0.377126 | 0.756810 | 3.704 / 0.000 / 0.948864 |
| resize_640 | 14.03 / 14.13 | 17.08 / 17.54 | 6.01 / 2.70 | 0.605852 | 0.831987 | 1.466 / 0.867 / 0.977273 |

## Reproduction and limits

- Scripts: `scripts/speedup_{train,eval,bundle,export,compress,coreml,tracking,video}.py`; `speedup_collect.sh` transfers frozen checkpoints, and `speedup_finalize_mini.py` chooses compression candidates without held-out scores.
- PC weights: `~/gro/repo/runs/fretnet/speedup_*/best.pt`; Mini exports: `~/gro-coreml/speedup/`. Per-checkpoint protocols/logs and selection hashes are under `eval/speedup/`; raw images and large checkpoints stay on the machines.
- CoreML compression uses coremltools 9.0: symmetric int8 linear weights, k-means palettes, and activation calibration using 32 deterministic VAL-only frames. [Apple documentation](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) explains why compression speedups must be measured on the target hardware.
- All new backbones/weights are torchvision BSD-3-Clause with the existing ImageNet caveat; see `LICENSES.md`. No Ultralytics code/weights enter this work.
- PC full test suite: 103 passed (including backbone checkpoint/export and synthetic homography tests). New Python scripts pass Ruff and compile checks; collection shell passes `bash -n`.
- Short training budgets are not convergence claims. Validation uses pipeline labels; eval_v2 contains pseudo-GT plus hand corrections, while 09-06 uses hand GT.


## CoreML parity against matching PyTorch checkpoints (VAL, 32 frames)

Coordinates are source-image pixels. Fit disagreement counts either runtime lacking a fit; it is not necessarily a fit lost by CoreML. Raw and refitted predictions are compared separately.

| Candidate / units | Raw keypoint mean / p95 px | Refit keypoint mean / p95 px | Refit wire mean / p95 px | Refit anchor flips/frame | Fit disagreements |
|---|---:|---:|---:|---:|---:|
| r832_int8 / CPU_AND_NE | 2.772 / 6.043 | 3.067 / 6.406 | 2.511 / 10.247 | 4.656 | 0 |
| r832_int8 / ALL | 2.749 / 5.875 | 3.028 / 6.435 | 2.477 / 10.227 | 4.344 | 0 |
| resize_416 / CPU_AND_NE | 0.404 / 0.855 | 0.455 / 0.843 | 4.074 / 18.074 | 1.531 | 0 |
| resize_416 / ALL | 0.422 / 0.877 | 0.647 / 0.948 | 3.900 / 16.792 | 2.125 | 0 |
| resize_512 / CPU_AND_NE | 2.188 / 4.392 | 1.966 / 5.713 | 6.255 / 26.831 | 5.438 | 0 |
| resize_512 / ALL | 0.385 / 0.860 | 0.938 / 0.972 | 3.698 / 16.062 | 2.844 | 0 |
| resize_640 / CPU_AND_NE | 0.399 / 0.813 | 0.473 / 0.850 | 0.632 / 3.201 | 1.781 | 0 |
| resize_640 / ALL | 0.403 / 0.831 | 0.442 / 0.838 | 0.651 / 3.497 | 1.531 | 0 |