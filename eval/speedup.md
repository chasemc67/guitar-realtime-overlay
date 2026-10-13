# Speeding up v4q without losing numbering accuracy

No Jam Dashboard app changes, release actions, or git commits. Experiments use the
PC RTX 4090 and the Mac Mini M4. Full protocol: [plan.json](speedup/plan.json).
This report is regenerated as results land; missing candidates are still pending.

Targets: CPU_AND_NE median approximately 8 ms or less, eval_v2 numbering >0.601,
and 09-06 numbering 1.000. Historical v1: 4.25 ms, numbering 0.601 / 1.000;
v4q-832 + 0.30 refit: 28.05 / 28.69 ms, numbering 0.714223 / 1.000,
eval_v2 D13 5.2753 / 2.1632 px, coverage 0.846059;
09-06 D13 0.7621 px, coverage 0.987216. These are prior measurements.

## Compression correctness finding

Initial default `r832_int8`, `r832_palette6`, and `r832_palette4` variants compressed
all eligible constants. A graph audit found this included `gx`/`gy` coordinate grids
and positional constants, not only learned weights. Those results are retained.
The distinct `r832_weights_*` variants and subsequent student/fine-tune compression
restrict compression to convolution/linear weights, preserving geometric constants.
This is an implementation correction based on the exported graph; thresholds and
bit widths are not retuned against held-out scores. See
[graph audit](speedup/coordinate_compression_audit.json). Later runs perform VAL
parity before held-out scoring. Compression reports record the exact scope.

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
| distill_efficientnet_b0_640 | 8.13 / 8.22 | 10.25 / 10.54 | 10.43 / 5.85 | 0.410848 | 0.791198 | 1.967 / 0.867 / 0.964489 |
| distill_mobilenet_v3_large_640 | 6.48 / 6.70 | 8.57 / 8.88 | 10.93 / 7.00 | 0.525938 | 0.788797 | 2.456 / 0.600 / 0.907670 |
| distill_resnet18_640 | 6.83 / 6.91 | 8.78 / 9.34 | 12.06 / 8.04 | 0.486329 | 0.771012 | 2.043 / 0.800 / 0.971591 |
| distill_resnet18_640_int8 | 6.91 / 7.13 | 11.80 / 12.80 | 13.08 / 7.66 | 0.436424 | 0.769688 | 2.214 / 0.800 / 0.958807 |
| distill_resnet18_640_palette4 | 6.89 / 7.16 | 9.30 / 10.83 | 15.94 / 10.26 | 0.503342 | 0.743564 | 2.443 / 0.467 / 0.982955 |
| distill_resnet18_640_palette6 | 6.89 / 7.27 | 9.14 / 10.13 | 11.70 / 7.14 | 0.518841 | 0.777463 | 2.015 / 0.733 / 0.970170 |
| distill_resnet18_640_w8a8 | 5.53 / 5.71 | 5.58 / 5.92 | 12.11 / 7.80 | 0.514572 | 0.775614 | 2.189 / 0.733 / 0.968750 |
| distill_resnet34_640 | 9.00 / 9.69 | 10.83 / 11.36 | 10.10 / 2.47 | 0.613685 | 0.787168 | 1.332 / 0.867 / 0.971591 |
| distill_resnet34_640_int8 | 8.73 / 8.81 | 13.61 / 14.04 | 10.31 / 3.62 | 0.646030 | 0.783967 | 1.309 / 0.933 / 0.968750 |
| distill_resnet34_640_palette4 | 8.86 / 8.96 | 10.89 / 11.36 | 8.11 / 2.53 | 0.624875 | 0.809278 | 1.824 / 0.467 / 0.901989 |
| distill_resnet34_640_palette6 | 8.79 / 8.88 | 10.80 / 11.29 | 13.69 / 9.23 | 0.623451 | 0.785420 | 1.247 / 0.867 / 0.974432 |
| distill_resnet34_640_w8a8 | 6.51 / 6.88 | 6.50 / 6.89 | 9.37 / 2.70 | 0.637957 | 0.788574 | 1.381 / 0.867 / 0.974432 |
| ft_416 | 5.53 / 5.61 | 8.21 / 8.52 | 9.85 / 5.56 | 0.375850 | 0.815923 | 2.606 / 0.733 / 0.965909 |
| ft_512 | 7.96 / 8.11 | 10.30 / 10.54 | 7.01 / 3.58 | 0.526869 | 0.827605 | 1.905 / 0.800 / 0.984375 |
| ft_640 | 14.09 / 14.48 | 17.09 / 17.59 | 6.25 / 2.36 | 0.637908 | 0.834601 | 1.746 / 0.867 / 0.980114 |
| r832_fp16 | 26.76 / 26.90 | 30.80 / 31.18 | 5.38 / 2.14 | 0.692329 | 0.844557 | 0.792 / 1.000 / 0.987216 |
| r832_int8 | 27.02 / 27.30 | 37.06 / 38.60 | 6.40 / 2.92 | 0.665034 | 0.807718 | 1.705 / 0.933 / 0.947443 |
| r832_palette4 | 24.82 / 24.99 | 28.66 / 29.10 | 98.48 / 61.53 | 0.022529 | 0.152562 | 140.285 / 0.000 / 0.161932 |
| r832_palette6 | 24.85 / 25.09 | 28.69 / 28.99 | 16.55 / 7.65 | 0.198906 | 0.600604 | 12.795 / 0.067 / 0.615057 |
| r832_w8a8 | 20.12 / 20.28 | 20.16 / 20.39 | 5.13 / 2.18 | 0.703234 | 0.842280 | 0.801 / 1.000 / 0.984375 |
| r832_weights_int8 | 27.89 / 28.67 | 37.46 / 38.96 | 4.65 / 2.08 | 0.722823 | 0.854262 | 0.785 / 1.000 / 0.985795 |
| r832_weights_palette4 | 24.89 / 25.06 | 28.64 / 28.85 | 5.98 / 2.70 | 0.761368 | 0.735088 | 2.813 / 0.867 / 0.723011 |
| r832_weights_palette6 | 24.87 / 25.95 | 28.68 / 29.51 | 4.53 / 2.09 | 0.735140 | 0.844244 | 0.823 / 1.000 / 0.982955 |
| resize_416 | 5.49 / 5.58 | 8.32 / 8.56 | 19.66 / 10.04 | 0.116071 | 0.646193 | 9.217 / 0.000 / 0.839489 |
| resize_512 | 8.09 / 8.23 | 10.08 / 10.56 | 9.21 / 5.17 | 0.381345 | 0.771110 | 3.923 / 0.000 / 0.940341 |
| resize_640 | 14.03 / 14.13 | 17.08 / 17.54 | 6.24 / 2.75 | 0.614311 | 0.830186 | 1.465 / 0.867 / 0.977273 |

## Recommendations

No collected CoreML candidate yet meets all three strict targets (≤8.00 ms, >0.601, 1.000).
Fastest numbering-qualified candidate: **r832_w8a8**, 20.12 ms. No model is promoted by this experiment.
Its eval_v2 wire error is 5.133 px mean / 2.177 px median with 0.842280 coverage; 09-06 error is 0.801 px with 0.984375 coverage. Numbering alone is not evidence that every accuracy metric improved.

## PyTorch held-out results

These are separate runtime variants, scored once after VAL selection. Do not treat their accuracy as CoreML accuracy.

| Candidate | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|---:|
| distill_efficientnet_b0_640 | 11.459 / 6.056 | 0.406448 | 0.790906 | 2.000 / 0.867 / 0.965909 |
| distill_mobilenet_v3_large_640 | 10.543 / 6.229 | 0.497645 | 0.787996 | 2.490 / 0.533 / 0.904830 |
| distill_resnet18_640 | 12.239 / 7.865 | 0.511465 | 0.775591 | 2.107 / 0.800 / 0.970170 |
| distill_resnet34_640 | 9.956 / 4.213 | 0.646410 | 0.791918 | 1.334 / 0.867 / 0.971591 |
| ft_416 | 10.137 / 5.815 | 0.434524 | 0.824281 | 2.618 / 0.800 / 0.965909 |
| ft_512 | 6.645 / 3.445 | 0.504644 | 0.830226 | 1.857 / 0.867 / 0.985795 |
| ft_640 | 6.159 / 2.276 | 0.625314 | 0.836450 | 1.744 / 0.867 / 0.980114 |
| resize_416 | 17.864 / 10.056 | 0.116071 | 0.643467 | 9.009 / 0.000 / 0.825284 |
| resize_512 | 9.024 / 4.993 | 0.363095 | 0.758734 | 3.889 / 0.000 / 0.954545 |
| resize_640 | 6.151 / 2.656 | 0.598889 | 0.829694 | 1.464 / 0.800 / 0.975852 |

## VAL selection

All refit sweeps use 1,498 VAL images and thresholds 0.30–0.60; choose numbering, then coverage, then lower D13. Every checkpoint is selected on VAL mean keypoint error. Filename guards exclude eval_v2 IDs and 09-06 from train/VAL. 640 was selected for distillation from the resize VAL sweep before held-out scoring. All four students use eight epochs, GT + teacher heatmap/keypoint/query-kernel/confidence losses; fine-tunes use four epochs and GT losses. Separate extended_512 and extended_640 continuations select the backbone using VAL numbering only, then each independently train 20 additional epochs from the same 640 student checkpoint, with the same 832 teacher. Compression candidates use VAL numbering plus latency only (see plan).

| Candidate | Selected threshold | VAL numbering | VAL coverage | VAL D13 mean px |
|---|---:|---:|---:|---:|
| resize_416 | 0.30 | 0.357143 | 0.753285 | 8.659 |
| resize_512 | 0.30 | 0.456914 | 0.799323 | 5.600 |
| resize_640 | 0.30 | 0.750334 | 0.838868 | 3.091 |
| distill_efficientnet_b0_640 | 0.30 | 0.618158 | 0.833188 | 4.116 |
| distill_mobilenet_v3_large_640 | 0.30 | 0.635514 | 0.848263 | 3.639 |
| distill_resnet18_640 | 0.30 | 0.693591 | 0.879366 | 3.452 |
| distill_resnet34_640 | 0.30 | 0.826435 | 0.843082 | 2.500 |
| ft_416 | 0.30 | 0.390521 | 0.860654 | 5.727 |
| ft_512 | 0.30 | 0.608812 | 0.839836 | 3.916 |
| ft_640 | 0.30 | 0.776368 | 0.862246 | 2.780 |

### Training checkpoint selection

Each continuation starts from the original student's best checkpoint, not the other continuation. Errors here are validation keypoint errors in network-input pixels, not held-out wire errors.

| Candidate | Completed epochs | Selected epoch | VAL mean / median / p95 keypoint error px |
|---|---:|---:|---:|
| distill_efficientnet_b0_640 | 8 | 8 | 3.862 / 1.378 / 13.294 |
| distill_mobilenet_v3_large_640 | 8 | 8 | 3.624 / 1.427 / 11.304 |
| distill_resnet18_640 | 8 | 4 | 3.375 / 1.275 / 11.757 |
| distill_resnet34_640 | 8 | 6 | 2.667 / 1.065 / 8.109 |
| ft_416 | 4 | 3 | 2.591 / 1.268 / 8.188 |
| ft_512 | 4 | 4 | 2.797 / 1.117 / 8.996 |
| ft_640 | 4 | 1 | 3.077 / 1.054 / 10.337 |

## Tracking

Each sparse labelled target is paired with its actual preceding decoded video frame. Inference/refit on that predecessor is propagated by LK optical flow and a RANSAC homography. This reports **skipped-frame-only accuracy**, not accuracy of a densely labelled full video. Effective adjacent-pair time is (inference + preprocess + Python refit + tracking) / 2, excluding decode. It includes Python postprocessing overhead and is not a native app FPS claim. Window 21/31 is selected on VAL synthetic translations before held-out frames. The original 832 control uses native-image flow; subsequent optimized variants also select long edge 320/640/native, choosing the fastest VAL configuration within a fixed 1-source-pixel mean-error budget. Four eval_v2 targets at frame zero have no predecessor and are excluded, leaving 228 eval_v2 and all 15 09-06 targets (explicit IDs in JSON). The first baseline attempt stopped at the first such target before scoring any held-out frame; the corrected run reused its frozen VAL selection.

| Candidate | Effective pair median / p95 ms per frame | Flow median / p95 ms | eval_v2 D13 mean / median px | Numbering / coverage | 09-06 D13 / numbering / coverage | Flow failures |
|---|---:|---:|---:|---:|---:|---:|
| distill_resnet18_640 | 7.512 / 9.493 | 3.108 / 6.186 | 12.513 / 7.240 | 0.482345 / 0.778377 | 2.082 / 0.733 / 0.961648 | 2 |
| r832 | 21.971 / 24.648 | 13.518 / 14.784 | 5.152 / 2.064 | 0.732525 / 0.837441 | 0.797 / 1.000 / 0.988636 | 0 |
| r832_scaled | 16.947 / 18.906 | 3.036 / 5.860 | 5.404 / 2.187 | 0.726314 / 0.837441 | 0.797 / 1.000 / 0.988636 | 0 |

Contiguous-video replay uses alternating inference/flow on up to 120 frames each from 09-06, nBpe2 and Stan. There is no dense ground truth; model-to-model agreement in the raw reports is **not** an accuracy measurement.

| Candidate / clip | Frames | Actual alternating pair median / p95 ms per frame | Flow failures |
|---|---:|---:|---:|
| distill_resnet18_640 / 2026-09-06_09-00-01 | 120 | 6.822 / 8.436 | 0 |
| distill_resnet18_640 / yt_nBpe2YQEzZo | 120 | 7.438 / 8.607 | 0 |
| distill_resnet18_640 / stan_head3s | 90 | 7.288 / 8.779 | 0 |
| r832 / 2026-09-06_09-00-01 | 120 | 19.255 / 20.500 | 0 |
| r832 / yt_nBpe2YQEzZo | 120 | 24.951 / 26.522 | 0 |
| r832 / stan_head3s | 90 | 23.471 / 24.987 | 0 |
| r832_scaled / 2026-09-06_09-00-01 | 120 | 16.130 / 17.503 | 0 |
| r832_scaled / yt_nBpe2YQEzZo | 120 | 17.633 / 19.324 | 0 |
| r832_scaled / stan_head3s | 90 | 16.867 / 17.837 | 0 |

## ALL compute-mode accuracy

| Candidate | CPU_AND_NE median / p95 ms | ALL median / p95 ms | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|---:|---:|---:|
| distill_efficientnet_b0_640 | 8.13 / 8.22 | 10.25 / 10.54 | 10.22 / 6.14 | 0.502227 | 0.793883 | 1.953 / 0.867 / 0.965909 |
| distill_mobilenet_v3_large_640 | 6.48 / 6.70 | 8.57 / 8.88 | 11.16 / 6.73 | 0.424123 | 0.790070 | 2.496 / 0.600 / 0.899148 |
| distill_resnet18_640 | 6.83 / 6.91 | 8.78 / 9.34 | 12.95 / 7.39 | 0.428280 | 0.764584 | 2.010 / 0.733 / 0.967330 |
| distill_resnet18_640_int8 | 6.91 / 7.13 | 11.80 / 12.80 | 12.23 / 7.80 | 0.535539 | 0.781910 | 2.159 / 0.800 / 0.974432 |
| distill_resnet18_640_palette4 | 6.89 / 7.16 | 9.30 / 10.83 | 16.12 / 10.29 | 0.427629 | 0.747941 | 2.396 / 0.533 / 0.974432 |
| distill_resnet18_640_palette6 | 6.89 / 7.27 | 9.14 / 10.13 | 11.67 / 6.74 | 0.475275 | 0.773988 | 1.847 / 0.867 / 0.961648 |
| distill_resnet18_640_w8a8 | 5.53 / 5.71 | 5.58 / 5.92 | 12.11 / 7.80 | 0.514572 | 0.775614 | 2.189 / 0.733 / 0.968750 |
| distill_resnet34_640 | 9.00 / 9.69 | 10.83 / 11.36 | 9.74 / 2.51 | 0.635143 | 0.785971 | 1.331 / 0.867 / 0.971591 |
| distill_resnet34_640_int8 | 8.73 / 8.81 | 13.61 / 14.04 | 10.32 / 3.52 | 0.627361 | 0.786751 | 1.319 / 0.933 / 0.968750 |
| distill_resnet34_640_palette4 | 8.86 / 8.96 | 10.89 / 11.36 | 8.03 / 2.71 | 0.635885 | 0.813567 | 1.848 / 0.467 / 0.901989 |
| distill_resnet34_640_palette6 | 8.79 / 8.88 | 10.80 / 11.29 | 11.70 / 8.32 | 0.640392 | 0.791079 | 1.253 / 0.933 / 0.973011 |
| distill_resnet34_640_w8a8 | 6.51 / 6.88 | 6.50 / 6.89 | 9.37 / 2.70 | 0.637957 | 0.788574 | 1.381 / 0.867 / 0.974432 |
| ft_416 | 5.53 / 5.61 | 8.21 / 8.52 | 9.72 / 5.52 | 0.445153 | 0.818751 | 2.612 / 0.733 / 0.965909 |
| ft_512 | 7.96 / 8.11 | 10.30 / 10.54 | 7.11 / 3.51 | 0.523646 | 0.828343 | 1.859 / 0.867 / 0.985795 |
| ft_640 | 14.09 / 14.48 | 17.09 / 17.59 | 6.21 / 2.35 | 0.617925 | 0.832415 | 1.741 / 0.867 / 0.980114 |
| r832_fp16 | 26.76 / 26.90 | 30.80 / 31.18 | 5.39 / 2.12 | 0.727668 | 0.840123 | 0.784 / 1.000 / 0.987216 |
| r832_int8 | 27.02 / 27.30 | 37.06 / 38.60 | 5.95 / 2.85 | 0.637708 | 0.808190 | 1.715 / 0.933 / 0.953125 |
| r832_palette4 | 24.82 / 24.99 | 28.66 / 29.10 | 105.01 / 67.91 | 0.025626 | 0.148298 | 125.499 / 0.000 / 0.140625 |
| r832_palette6 | 24.85 / 25.09 | 28.69 / 28.99 | 14.53 / 7.56 | 0.170460 | 0.616362 | 11.418 / 0.000 / 0.653409 |
| r832_w8a8 | 20.12 / 20.28 | 20.16 / 20.39 | 5.13 / 2.18 | 0.703234 | 0.842280 | 0.801 / 1.000 / 0.984375 |
| r832_weights_int8 | 27.89 / 28.67 | 37.46 / 38.96 | 4.93 / 2.09 | 0.730316 | 0.852137 | 0.779 / 1.000 / 0.984375 |
| r832_weights_palette4 | 24.89 / 25.06 | 28.64 / 28.85 | 5.93 / 2.35 | 0.753517 | 0.728909 | 2.180 / 0.867 / 0.730114 |
| r832_weights_palette6 | 24.87 / 25.95 | 28.68 / 29.51 | 4.83 / 2.08 | 0.728646 | 0.847243 | 0.813 / 1.000 / 0.982955 |
| resize_416 | 5.49 / 5.58 | 8.32 / 8.56 | 17.97 / 9.45 | 0.255952 | 0.638830 | 8.945 / 0.000 / 0.822443 |
| resize_512 | 8.09 / 8.23 | 10.08 / 10.56 | 9.53 / 5.24 | 0.377126 | 0.756810 | 3.704 / 0.000 / 0.948864 |
| resize_640 | 14.03 / 14.13 | 17.08 / 17.54 | 6.01 / 2.70 | 0.605852 | 0.831987 | 1.466 / 0.867 / 0.977273 |

## Reproduction and limits

- Scripts: `scripts/speedup_{train,eval,bundle,export,compress,coreml,tracking,video}.py`; `speedup_collect.sh` transfers frozen checkpoints, and `speedup_finalize_mini.py` chooses compression candidates without held-out scores.
- PC weights: `~/gro/repo/runs/fretnet/speedup_*/best.pt`; Mini exports: `~/gro-coreml/speedup/`. Per-checkpoint protocols/logs and selection hashes are under `eval/speedup/`; raw images and large checkpoints stay on the machines.
- CoreML compression uses coremltools 9.0: symmetric int8 linear weights, k-means palettes, and activation calibration using 32 deterministic VAL-only frames. [Apple documentation](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) explains why compression speedups must be measured on the target hardware.
- All new backbones/weights are torchvision BSD-3-Clause with the existing ImageNet caveat; see `LICENSES.md`. No Ultralytics code/weights enter this work.
- PC full test suite: 104 passed (including backbone checkpoint/export and synthetic homography tests). New Python scripts pass Ruff and compile checks; collection shell passes `bash -n`.
- Short training budgets are not convergence claims. Validation uses pipeline labels; eval_v2 contains pseudo-GT plus hand corrections, while 09-06 uses hand GT.
- [PC environment and pretrained-weight hashes](speedup/pc_environment.json), [Mini environment](speedup/mini_environment.txt), [held-out manifest](speedup/bundle_manifest.json), and [VAL calibration manifest](speedup/calibration_manifest.json) identify this run. Training augmentation is not bitwise reproducible; checkpoint hashes identify the frozen evaluated models.
- The 09-06 result covers only 15 labelled frames. All conclusions are specific to these test sets and this Mini. A failed dense-video flow update reuses the previous fit; failure counts are reported, not hidden.


## CoreML parity against matching PyTorch checkpoints (VAL, 32 frames)

Coordinates are source-image pixels. Fit disagreement counts either runtime lacking a fit; it is not necessarily a fit lost by CoreML. Raw and refitted predictions are compared separately.

| Candidate / units | Raw keypoint mean / p95 px | Refit keypoint mean / p95 px | Refit wire mean / p95 px | Refit anchor flips/frame | Fit disagreements |
|---|---:|---:|---:|---:|---:|
| distill_efficientnet_b0_640 / CPU_AND_NE | 0.496 / 1.046 | 1.216 / 1.389 | 2.272 / 11.585 | 3.156 | 0 |
| distill_efficientnet_b0_640 / ALL | 0.494 / 1.047 | 1.201 / 1.421 | 3.111 / 15.256 | 4.531 | 0 |
| distill_mobilenet_v3_large_640 / CPU_AND_NE | 0.401 / 0.853 | 0.637 / 0.971 | 1.349 / 7.385 | 1.844 | 0 |
| distill_mobilenet_v3_large_640 / ALL | 0.396 / 0.848 | 0.587 / 0.890 | 1.554 / 7.288 | 1.719 | 0 |
| distill_resnet18_640 / CPU_AND_NE | 0.417 / 0.852 | 0.408 / 0.853 | 0.477 / 1.780 | 0.875 | 0 |
| distill_resnet18_640 / ALL | 0.421 / 0.860 | 0.414 / 0.862 | 0.652 / 2.531 | 0.500 | 0 |
| distill_resnet18_640_int8 / CPU_AND_NE | 0.474 / 0.933 | 0.837 / 0.972 | 1.817 / 2.169 | 1.656 | 0 |
| distill_resnet18_640_int8 / ALL | 0.453 / 0.916 | 0.467 / 0.927 | 0.693 / 2.119 | 0.969 | 0 |
| distill_resnet18_640_palette4 / CPU_AND_NE | 2.131 / 3.405 | 1.365 / 2.881 | 2.217 / 6.116 | 3.500 | 0 |
| distill_resnet18_640_palette4 / ALL | 2.105 / 3.447 | 1.462 / 2.778 | 2.295 / 5.965 | 3.562 | 0 |
| distill_resnet18_640_palette6 / CPU_AND_NE | 0.910 / 1.239 | 0.545 / 1.186 | 0.523 / 1.736 | 1.156 | 0 |
| distill_resnet18_640_palette6 / ALL | 0.962 / 1.225 | 0.615 / 1.221 | 0.728 / 2.838 | 1.062 | 0 |
| distill_resnet18_640_w8a8 / CPU_AND_NE | 0.770 / 1.175 | 0.575 / 1.097 | 0.959 / 3.615 | 1.500 | 0 |
| distill_resnet18_640_w8a8 / ALL | 0.770 / 1.175 | 0.575 / 1.097 | 0.959 / 3.615 | 1.500 | 0 |
| distill_resnet34_640 / CPU_AND_NE | 0.375 / 0.839 | 0.413 / 0.837 | 0.590 / 2.513 | 0.938 | 0 |
| distill_resnet34_640 / ALL | 0.383 / 0.830 | 0.378 / 0.824 | 0.487 / 1.862 | 0.562 | 0 |
| distill_resnet34_640_int8 / CPU_AND_NE | 0.394 / 0.854 | 0.413 / 0.841 | 0.360 / 0.928 | 1.062 | 0 |
| distill_resnet34_640_int8 / ALL | 0.402 / 0.871 | 0.482 / 0.872 | 0.729 / 4.007 | 1.094 | 0 |
| distill_resnet34_640_palette4 / CPU_AND_NE | 0.986 / 2.468 | 1.026 / 2.351 | 1.127 / 3.088 | 3.375 | 0 |
| distill_resnet34_640_palette4 / ALL | 0.975 / 2.502 | 1.019 / 2.494 | 0.999 / 4.017 | 3.688 | 0 |
| distill_resnet34_640_palette6 / CPU_AND_NE | 0.614 / 1.513 | 0.610 / 1.362 | 0.725 / 2.956 | 2.219 | 0 |
| distill_resnet34_640_palette6 / ALL | 0.609 / 1.499 | 0.696 / 1.433 | 0.803 / 3.785 | 2.312 | 0 |
| distill_resnet34_640_w8a8 / CPU_AND_NE | 0.466 / 1.019 | 0.484 / 1.007 | 0.559 / 2.322 | 1.344 | 0 |
| distill_resnet34_640_w8a8 / ALL | 0.466 / 1.019 | 0.484 / 1.007 | 0.559 / 2.322 | 1.344 | 0 |
| ft_416 / CPU_AND_NE | 0.444 / 0.887 | 1.859 / 1.112 | 4.941 / 28.681 | 2.375 | 0 |
| ft_416 / ALL | 0.406 / 0.887 | 0.668 / 0.964 | 2.666 / 12.100 | 1.500 | 0 |
| ft_512 / CPU_AND_NE | 1.254 / 2.643 | 1.267 / 2.408 | 2.670 / 6.313 | 3.594 | 0 |
| ft_512 / ALL | 0.402 / 0.890 | 0.624 / 0.939 | 1.199 / 5.398 | 2.094 | 0 |
| ft_640 / CPU_AND_NE | 0.381 / 0.836 | 0.474 / 0.856 | 0.556 / 2.927 | 1.812 | 0 |
| ft_640 / ALL | 0.370 / 0.816 | 0.451 / 0.833 | 0.577 / 2.254 | 1.031 | 0 |
| r832_fp16 / CPU_AND_NE | 0.384 / 0.843 | 1.168 / 0.970 | 2.812 / 9.987 | 1.406 | 0 |
| r832_fp16 / ALL | 0.676 / 1.503 | 0.747 / 1.512 | 0.736 / 3.336 | 2.094 | 0 |
| r832_int8 / CPU_AND_NE | 2.772 / 6.043 | 3.067 / 6.406 | 2.511 / 10.247 | 4.656 | 0 |
| r832_int8 / ALL | 2.749 / 5.875 | 3.028 / 6.435 | 2.477 / 10.227 | 4.344 | 0 |
| r832_palette4 / CPU_AND_NE | 47.392 / 145.959 | 117.091 / 394.387 | 655.092 / 2533.285 | 32.562 | 4 |
| r832_palette4 / ALL | 47.106 / 144.949 | 126.096 / 411.144 | 445.582 / 1080.157 | 33.406 | 10 |
| r832_palette6 / CPU_AND_NE | 8.572 / 12.689 | 15.538 / 47.880 | 249.504 / 436.720 | 16.906 | 1 |
| r832_palette6 / ALL | 8.536 / 12.689 | 17.860 / 58.871 | 49.204 / 195.002 | 16.719 | 0 |
| r832_w8a8 / CPU_AND_NE | 1.263 / 2.928 | 1.005 / 2.158 | 1.000 / 3.717 | 3.531 | 0 |
| r832_w8a8 / ALL | 1.263 / 2.928 | 1.005 / 2.158 | 1.000 / 3.717 | 3.531 | 0 |
| r832_weights_int8 / CPU_AND_NE | 0.692 / 1.252 | 0.833 / 1.245 | 0.900 / 3.736 | 2.625 | 0 |
| r832_weights_int8 / ALL | 0.673 / 1.206 | 0.634 / 1.181 | 0.642 / 2.822 | 1.969 | 0 |
| r832_weights_palette4 / CPU_AND_NE | 23.494 / 131.172 | 15.308 / 93.337 | 126.866 / 568.839 | 12.219 | 0 |
| r832_weights_palette4 / ALL | 23.524 / 126.661 | 11.768 / 53.709 | 56.765 / 394.784 | 12.000 | 0 |
| r832_weights_palette6 / CPU_AND_NE | 2.555 / 5.216 | 1.327 / 4.341 | 1.146 / 4.069 | 4.219 | 0 |
| r832_weights_palette6 / ALL | 2.546 / 5.466 | 1.511 / 4.794 | 1.859 / 7.431 | 4.031 | 0 |
| resize_416 / CPU_AND_NE | 0.404 / 0.855 | 0.455 / 0.843 | 4.074 / 18.074 | 1.531 | 0 |
| resize_416 / ALL | 0.422 / 0.877 | 0.647 / 0.948 | 3.900 / 16.792 | 2.125 | 0 |
| resize_512 / CPU_AND_NE | 2.188 / 4.392 | 1.966 / 5.713 | 6.255 / 26.831 | 5.438 | 0 |
| resize_512 / ALL | 0.385 / 0.860 | 0.938 / 0.972 | 3.698 / 16.062 | 2.844 | 0 |
| resize_640 / CPU_AND_NE | 0.399 / 0.813 | 0.473 / 0.850 | 0.632 / 3.201 | 1.781 | 0 |
| resize_640 / ALL | 0.403 / 0.831 | 0.442 / 0.838 | 0.651 / 3.497 | 1.531 | 0 |

## Direct-inference controls on the same tracking targets

Aggregated from saved direct CoreML predictions on exactly the 228 + 15 tracking targets. No additional model evaluation or tuning is performed.

| Tracking variant / direct reference | Direct eval_v2 D13 mean / median | Direct numbering / coverage | Direct 09-06 D13 / numbering / coverage |
|---|---:|---:|---:|
| distill_resnet18_640 / distill_resnet18_640 | 12.071 / 8.005 | 0.485812 / 0.772184 | 2.043 / 0.800 / 0.971591 |
| r832 / r832_fp16 | 5.350 / 2.126 | 0.696476 / 0.844767 | 0.792 / 1.000 / 0.987216 |
| r832_scaled / r832_fp16 | 5.350 / 2.126 | 0.696476 / 0.844767 | 0.792 / 1.000 / 0.987216 |