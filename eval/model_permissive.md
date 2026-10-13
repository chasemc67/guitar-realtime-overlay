# Model v3 (FretNet): permissively licensed fretboard keypoints vs v1, 2026-10-12

v1 (YOLO11s-pose) is AGPL-3.0 through Ultralytics' code and COCO weights. v3 is a replacement candidate with no
Ultralytics code or weights anywhere in its training, export or inference path. Licences: [`LICENSES.md`](../LICENSES.md).

**Result:** v3 is shipped in the Jam Dashboard desktop app next to v1, and the app has a model picker. v1 stays the
default because v3 numbers frets clearly worse on eval v2. v3 places wires better, and it finds more of them.

## Model

- **Code** (ours): `src/gro/fretnet/`. Training is `scripts/train_fretnet.py`, export is `scripts/export_fretnet.py`.
  `gro.infer.FretNetBackend` / `load_backend` pick it up automatically from a `.pt` checkpoint.
- **Backbone:** torchvision ResNet-50 with ImageNet `IMAGENET1K_V2` init (BSD-3-Clause).
- **Neck:** an FPN that merges strides 4 to 32 into one 128-channel stride-4 map (160×160 at 640).
- **Head:** 50 keypoint heatmaps plus 50 confidence maps.
  - Location is a soft-argmax inside a 5×5-cell window around the peak. A global soft-argmax would average
    neighbouring wires, because frets are periodic.
  - Confidence is the confidence map read under the heatmap's softmax. That is local evidence, like YOLO's per-anchor
    keypoint confidence. A first version pooled global features, and its val coverage fell from 0.92 to 0.81 while
    train loss kept dropping (run `v3_r50`, stopped at epoch 15).
- **Input / output:** same as v1. 640×640 letterbox, RGB in [0, 1] (CoreML: image input with 1/255 baked in). The
  output is v1's raw head layout `(1, 155, A)` with a single anchor (A = 1), so every v1 decoder (Python
  `gro.infer.decode`, Swift `FretCore.decode`) and all post-processing work unchanged.
- **fp16:** the peak cell comes from an equality one-hot, not a flat argmax index. On the Neural Engine, CoreML runs
  fp16, which cannot represent an index of up to 25599 exactly. The first fp16 export was off by 8.5 px mean.
- **Data:** scale-v1aug (the v2_aug data: scale-v1 plus synthetic fret-wrap and occlusion copies; 5798 train, 1498 val),
  same splits as v1/v2. The YOLO-pose label text files are read by our own loader.
- **Augmentation:** our own random affine (scale ±25 %, rotation ±10°, translation ±10 %), HSV jitter, blur, random
  erasing. No horizontal flip (D1).
- **Training:** 50 epochs, AdamW lr 1e-3 (backbone ×0.5), cosine schedule, bf16, batch 32, about 45 s/epoch on the
  4090.
- **Loss:** cross-entropy to a Gaussian target (σ = 1 cell), L1 of the windowed soft-argmax, and BCE on the
  confidence.

| Run | Change | Epochs | Val err mean / median px @640 |
|---|---|---:|---|
| v3_r50 | global-pooled confidence head | 15 (stopped) | 3.25 / 1.22 (coverage falling) |
| **v3b_r50 (shipped as v3)** | local confidence map | 50 | 3.05 / 1.13 |
| v3c_r50nb | v3b plus a neighbour-fret hinge loss (`--neighbor-margin 1`) on the heatmap logit at the GT of fret ±1 | 50 (best = ep 35) | 3.09 / 1.10 |

## Eval v2 + 09-06 honest set (`scripts/eval_v2.py`, same frozen set and rules as `eval/model_v2_report.md`)

v1 and v2_aug go through ultralytics' predict; v3 goes through `gro.infer`, the tracker's own path. Full tables:
`eval/v2/v3b_r50.md`, `eval/v2/v3c_r50nb.md`.

| Model | eval v2 D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection | 09-06 D13 px | 09-06 numbering | 09-06 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 (shipped default) | 8.44 / 5.80 | 26.2 | 5.91 | **0.601** | 0.805 | 0.835 | **0.82** | **1.000** | 0.949 |
| v2_aug (AGPL, not shipped) | 6.27 / 4.42 | **17.5** | 5.78 | 0.692 | 0.960 | 0.997 | 1.04 | 1.000 | 0.915 |
| **v3b_r50 = v3** | **6.08 / 2.89** | 23.1 | 4.03 | 0.367 | 0.907 | **1.000** | 1.11 | 0.867 | **0.990** |
| v3c_r50nb | 6.61 / **2.48** | 25.7 | **3.66** | 0.377 | 0.894 | 1.000 | 1.54 | 0.733 | 0.982 |

Per guitar (D13 mean px / numbering / coverage):

| Clip (guitar) | Frames | v1 | v3b | v3c |
|---|---:|---|---|---|
| yt_9oJLRxucpNI (Gretsch 6120) | 11 | 7.04 / 0.00 / 1.00 | 11.76 / 0.00 / 1.00 | 11.80 / 0.00 / 1.00 |
| yt_FbEe_5tGOFM (Gibson SG Special) | 24 | 1.05 / 1.00 / 0.98 | 5.36 / 0.00 / 0.95 | 9.91 / 0.00 / 0.85 |
| yt_cgYA0Xt417w (Gibson SG '61, steep close-up) | 14 | 44.71 / 0.14 / 0.77 | 2.27 / 0.79 / 0.94 | 5.08 / 0.36 / 0.94 |
| yt_GEljpsfLj8o (Les Paul goldtop) | 1 | 2.22 / 1.00 / 0.95 | 1.59 / 0.00 / 0.97 | 6.23 / 0.00 / 0.97 |
| yt_N_nayTl9HGY (Rickenbacker 360) | 13 | 3.78 / 0.08 / 0.57 | 5.67 / 0.00 / 0.82 | 2.98 / 0.00 / 0.85 |
| yt_ixOqhZfAjMg (PRS singlecut) | 24 | 2.38 / 0.79 / 0.98 | 5.12 / 0.04 / 0.91 | 4.28 / 0.29 / 0.82 |
| yt_j4D7XFUK7II (Jackson V) | 24 | 18.01 / 0.00 / 0.17 | 13.02 / 0.00 / 0.47 | 16.64 / 0.00 / 0.67 |
| yt_eM8DB4FT-jM (ESP George Lynch) | 24 | 2.87 / 0.75 / 0.99 | 3.85 / 0.50 / 0.97 | 3.37 / 0.25 / 0.95 |
| yt_yIYrqh7ulSk (Fender Jazzmaster) | 24 | 3.87 / 0.88 / 0.96 | 5.27 / 0.21 / 0.94 | 4.04 / 0.50 / 0.90 |
| yt_VT4VkOkFIRE (ESP Eclipse-style) | 3 | 3.64 / 1.00 / 0.94 | 1.33 / 1.00 / 0.98 | 1.08 / 1.00 / 0.99 |
| yt_nBpe2YQEzZo (Fender Strat) | 21 | 3.92 / 0.86 / 0.98 | 3.34 / 0.86 / 0.95 | 2.29 / 1.00 / 0.90 |
| yt_tNAX9maMla8 (Superstrat, no inlays) | 1 | n/a / 0.00 / 0.00 | 10.95 / 0.00 / 0.85 | 8.77 / 0.00 / 0.80 |
| yt_vXzydcRC1bA (ESP superstrat) | 24 | 13.69 / 0.92 / 1.00 | 13.27 / 0.92 / 0.99 | 14.12 / 0.92 / 0.93 |
| stan_june2 (Telecaster) | 24 | 2.54 / 1.00 / 0.99 | 2.27 / 0.83 / 0.96 | 1.91 / 0.96 / 0.95 |

## Reading

- **v3 aligns the grid better than v1.**
  - Macro D13 is 6.08 vs 8.44 px, median 2.89 vs 5.80.
  - Nearest-wire error is 4.03 vs 5.91 px.
  - Detection is 1.00 vs 0.835, and coverage 0.907 vs 0.805.
  - It fixes v1's steep SG '61 close-up (44.7 → 2.3 px, numbering 0.14 → 0.79).
- **v3 numbers frets clearly worse.** Numbering is 0.367 vs 0.601, and 0.867 vs 1.000 on 09-06. A frame fails
  numbering if any one prediction sits on the wrong GT wire. Most v3 failures are a few keypoints in an otherwise
  correct frame:
  - one side of a wire slips to the neighbouring wire (±1–2);
  - extra confident keypoints past the last fret, e.g. 22/23 on a 21/22-fret board (62 of 139 failing frames);
  - a few keypoints jump ±7–8 frets.

  The tracker's RANSAC fit rejects isolated slips. In the e2e runs (09-06, nBpe2), v3 and v1 settle on the same fret
  count and the overlays line up (screenshots in jam-dashboard `out/permissive/`). The metric, though, is what the
  promotion rule uses.
- **The neighbour-margin loss (v3c) does not fix it.** Numbering is 0.377 on eval v2 and worse on 09-06 (0.733).
  The val slip rate (keypoint nearer to a neighbouring fret's GT than to its own) stays about 5 %.
- **Decision:** ship v3 in the picker as the permissive option. Keep **v1 as the default**, because v3 is clearly
  worse on numbering, a headline metric.
- **Next:** attack numbering, not geometry. Options:
  - a coarse global fret-index cue, e.g. a regression branch that gates the heatmap window;
  - a learned "beyond the last fret" confidence (board-end negatives);
  - a fret-order consistency step in the head (monotone wires along the neck).

## CoreML (Mac Mini M4, fp16, `eval/v3/`)

Parity vs PyTorch, 90 frames from 09-06, Stan and nBpe2 (`scripts/coreml_parity.py`):

| Export | Keypoint px mean / p95 / max | Wire px mean / p95 / max | Anchor flips per frame | Fits lost |
|---|---|---|---:|---:|
| fp32 | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.001 | 0 | 0 |
| fp16 (shipped) | 0.53 / 1.16 / 124.5 | 1.02 / 4.21 / 35.1 | 0.011 | 0 |

The fp16 maximum is a rare near-tie between two heatmap peaks: one keypoint lands on another wire. RANSAC drops it.

Latency, model call, batch 1, 200 calls, median / p95 ms (`eval/v3/bench_v3b.json`):

| Model | ALL | CPU_AND_NE | CPU_AND_GPU | load (ALL) |
|---|---|---|---|---:|
| v1 fp16 (yolo11s) | 4.25 / 4.59 | 4.26 / 4.57 | 12.2 / 12.4 | 0.84 s |
| v3 fp16 (FretNet R50) | 12.7 / 13.1 | 12.7 / 12.9 | 26.9 / 27.4 | 3.2 s |

- **Swift helper** (`FretTrackerCLI --bench 200`, compute units all), total per frame: v1 3.8 ms, v3 12.3 ms.
- **In the app** (Live Fretboard, dev build): v3's model step takes 13–21 ms vs v1's 4–10 ms. It keeps up with
  30 fps video.
- **Backbone cost** on the ANE: ResNet-50 at 640 is about 10 ms of the 12.7 ms. ResNet-34 would be about 7.5 ms and
  ResNet-18 about 5.7 ms in total (untrained-weights export; accuracy not measured).
- **Size:** the CoreML package is 47 MB (v1: 22 MB).

## Round 2 (2026-10-12): numbering, permissive candidates vs v1

Goal: beat v1's eval v2 numbering of 0.601 without regressing wire error (v1 D13 8.44 / 5.80 px) or 09-06 (v1 D13
0.82 px, numbering 1.000, coverage 0.949).

### Why v3 numbering fails (`eval_model.metrics` now records the slot behind each index shift: `shift_keypoints`)

| Model | Failing frames | Wrong slots: ±1 / ±2 / larger jumps / beyond the last GT wire | Wrong slots at frets 0–8 / 9–14 / 15–23 | Whole-frame shifts |
|---|---:|---|---|---:|
| v1 | 79 of 232 | 384 / 90 / 62 / 30 | 101 / 157 / 308 | 2 (vXzydcRC1bA, +4) |
| v3b | 139 of 231 | 537 / 121 / 156 / 38 | 154 / 105 / 593 | 2 (same frames) |

- **v3 failures are scattered, single-keypoint slips.** A failing v3 frame has a median of 3 to 5 wrong slots out of
  about 40. Almost none are a consistent ±k shift of the whole board. v1 fails differently: it misses whole boards
  (detection 0.835) or shifts a run of wires (SG '61 close-up, Jackson V).
- **The slips cluster at frets 15–23.** That is where wires are 2 to 3 heatmap cells apart at 640, so the
  neighbouring wire's peak is a near-tie.
  - Two v3 clips also jump 7 to 10 frets: FbEe_5tGOFM (−7/−8) and j4D7XFUK7II (−9/−10). A single heatmap jumps to a
    look-alike wire far up the neck.
- **v1 does use the whole neck.** YOLO-pose regresses all 50 keypoints from one anchor's feature vector, which comes
  through C2PSA attention at stride 32. That makes the fret indices one joint decision, consistent but imprecise.
- **v3 does not.** Each of v3's 50 heatmaps is an independent 1×1 conv on local stride-4 features. Nothing ties fret
  k's peak to fret k±1's peak or to the inlay pattern.
- **Round-2 fix:** give every keypoint a global decision. Two heads:
  - `head="query"`: a transformer over the stride-32 tokens, plus 50 keypoint queries that attend to each other and
    to the image and add a dynamic term to each heatmap.
  - `SimCCNet`: per-keypoint tokens built from the whole stride-16 map, then 1D x/y classification.
- **Training data:** 1447 of 5798 train frames are labelled to fret 23 or higher. Round 2 oversamples them ×2.

- **The fret-order gate catches isolated slips but not runs.** `order_gate` is in the export head, so CoreML and the
  tracker see it. It drops a keypoint whose position along the neck breaks the equal-temperament spacing of its index
  neighbours (f±1, f±2). On v3b it lowers D13 (6.08 → 5.56) and raises numbering only slightly (0.367 → 0.389). Most
  remaining failures are runs: a stretch of high frets (e.g. 14–21 on the Gretsch, 21–22 on the SG's bass side) all
  shifted +1, or the SG's bass frets 4–6 placed on wires 11–14. Locally those runs are self-consistent; the model
  lost count where wires are dense. Counting needs global reasoning (the query / SimCC heads) or more pixels per
  fret.
- **The cross-side wire gate doesn't help numbering either.** `wire_gate` drops a wire whose two ends sit on
  different wires: the treble-minus-bass offset along the neck, from a robust line fit over frets, is more than half a
  fret gap off. Many v4q failures are one-sided runs (PRS bass 19–23 +1, Jazzmaster treble 16–19 −1). Numbering moves
  only 0.441 → 0.444, because a frame still fails on any one surviving wrong keypoint: unpaired ends past the board
  end, low frets 1–4 off by 2. Both gates cut D13, so they are worth keeping on whichever model ships.

### Candidates

All are permissive: torchvision code and ImageNet weights (BSD-3-Clause), own head code. "last" = final-epoch
checkpoint. Wire = eval v2 D13 mean / median px (macro).

| Candidate | Path | Licence | eval v2 wire px | Numbering | Coverage / detection | 09-06 D13 px | 09-06 numbering | 09-06 coverage |
|---|---|---|---|---:|---|---:|---:|---:|
| v1 (reference, AGPL) | — | AGPL-3.0 | 8.44 / 5.80 | 0.601 | 0.805 / 0.835 | 0.82 | 1.000 | 0.949 |
| v3b_r50 (shipped v3) | — | BSD-3 + own | 6.08 / 2.89 | 0.367 | 0.907 / 1.000 | 1.11 | 0.867 | 0.990 |
| v3c_r50nb last (ep 50) | b | BSD-3 + own | 7.53 / 2.62 | 0.406 | 0.914 / 1.000 | 1.78 | 0.800 | 0.986 |
| v3b + order gate (`order_gate`, no retraining) | b | BSD-3 + own | 5.56 / 2.86 | 0.389 | 0.885 / 1.000 | 1.11 | 0.867 | 0.990 |
| v4q_r50 best (val, ep 45): ResNet-50 + keypoint-query transformer (`--head query`), frets 23+ oversampled ×2, 50 ep | b | BSD-3 + own | 8.45 / 2.89 | 0.407 | 0.945 / 1.000 | 1.07 | 0.933 | 0.972 |
| v4q_r50 last (ep 50) | b | BSD-3 + own | 6.73 / 2.79 | 0.441 | 0.951 / 1.000 | 1.12 | 0.933 | 0.973 |
| v4q_r50 last + order gate | b | BSD-3 + own | 5.44 / 2.62 | 0.441 | 0.916 / 1.000 | 1.12 | 0.933 | 0.973 |
| v4q_r50 last + order + cross-side wire gate (`gate="wires"`) | b | BSD-3 + own | 5.32 / 2.52 | 0.444 | 0.901 / 1.000 | 1.12 | 0.933 | 0.973 |
| v3b + order + wire gate | b | BSD-3 + own | 5.21 / 2.81 | 0.389 | 0.870 / 1.000 | 1.11 | 0.867 | 0.990 |
| v5s_cnxt (best = last, ep 45): ConvNeXt-T + RTMPose-style SimCC head (own code), frets 23+ ×2 | a | BSD-3 + own | 18.14 / 16.41 | 0.207 | 0.946 / 1.000 | 3.80 | 0.533 | 0.987 |
| v4r_r50 (best = last, 30 ep fine-tune of v4q last): queries also regress x,y; Gaussian prior σ 8 px on the heatmap (`--head queryreg`) | b | BSD-3 + own | 9.91 / 6.51 | 0.431 | 0.942 / 1.000 | 1.17 | 0.933 | 0.984 |
| v4r_r50 prior σ 4 / 16 / off (eval-time sweep) | b | BSD-3 + own | 11.41 / 8.52 · 8.95 / 5.52 · 7.72 / 2.98 | 0.425 · 0.434 · 0.429 | 0.942 / 1.000 | 1.56 · 1.07 · 1.07 | 0.933 · 0.867 · 0.867 | 0.984 |
| **v4q_r50_832** (best = last, ep 45): v4q at 832×832 input, batch 24, scale aug 0.3, frets 23+ ×2 | b | BSD-3 + own | 7.22 / 2.56 | 0.522 | 0.917 / 1.000 | **0.78** | 0.933 | 0.990 |
| **v4q_r50_832 + order + wire gate** | b | BSD-3 + own | 5.64 / **2.05** | **0.554** | 0.854 / 1.000 | **0.78** | 0.933 | 0.990 |

### Round 2 reading

- **No candidate beats v1 on numbering, so nothing ships.** The closest is v4q_r50_832 with both gates:
  - numbering 0.554 vs 0.601;
  - wire error 5.64 / 2.05 px vs 8.44 / 5.80;
  - 09-06 D13 0.78 vs 0.82 px, but 09-06 numbering 0.933 vs 1.000.

  v1 stays the default. v3 stays the permissive option in the app. No CoreML export, app change or release this round.
- **What helped:**
  - **Resolution helped most:** v4q went from 0.441 at 640 to 0.522 at 832. Dense high-fret wires get more cells
    apart, and that is where v3/v4q lose count.
  - **Global attention helped modestly:** v3b's 0.367 rose to 0.441 with the keypoint-query transformer. Coverage
    also went up (0.907 → 0.951).
  - **The gates add about 0.03:** the fret-order and cross-side wire gates in the export head drop slipped
    keypoints. They always cut D13, at some coverage cost.
- **What didn't:**
  - **SimCC (path a, ConvNeXt-T) failed on new guitars:** 4.2 px on val but 18 px on eval v2.
  - **The query regression prior (v4r) didn't help.** Its coarse coordinates fit val (2.5 px) but not the held-out
    guitars.

  Both learn fully connected maps from image position to keypoint. On 14 unseen guitars they generalise worse than
  convolutional heatmaps. That makes the training set's limited guitar variety, not head capacity, the likely ceiling
  for absolute fret numbering.
- **Latency:** 832 input costs about 1.7× the 640 compute, roughly 21 ms on the M4 ANE by scaling v3's 12.7 ms. Not
  measured.
- **Next:**
  - v4q at 960 or a neck-crop second pass (resolution is the lever that works);
  - a ConvNeXt or R50 v4q at 832 trained longer;
  - more distinct guitars (21- and 24-fret boards, 480p footage like the SG Special), since generalisation, not fit,
    is the gap.

## Two-stage numbering experiments (2026-10-12)

Implementation: `src/gro/fretnet/numbering.py`, `scripts/train_numbering.py`,
`gro.infer.TwoStageBackend`, and `scripts/eval_v2.py --numbering CHECKPOINT`.
Stage 1 is `v4q_r50_832/last.pt`. Torchvision ImageNet ResNet-18 and MobileNetV3 Small
classifiers predict the numbering correction in −3..+3 from the whole rectified RGB neck. Pooling retains
16 longitudinal bins, so the head can use the inlay sequence and both ends of the board.
The strip covers fitted frets −2..26 and transverse coordinates −0.35..1.35.
Only the numbering slots change; observed pixel coordinates are preserved. A fixed 0.8
confidence threshold falls back to stage 1. No threshold is selected using eval_v2.

Training uses all 5798 scale-v1aug train / 1498 val images. Filename guards reject every
configured eval_v2 clip ID and the 09-06 recording date, including augmented copies.
GT homographies generate extended strips; random integer shifts, corner perturbations,
colour changes, blur and occlusion provide augmentation. Validation tests all seven
integer offsets. Classifier checkpoints are selected by validation offset accuracy only
(12 planned epochs, AdamW 2e-4, cosine decay, batch 4, bf16). The ResNet run completed 10/12 planned epochs
then was stopped for GPU contention; epoch 9 was selected by validation (0.890139).
MobileNet 1536 completed all 12 epochs; validation selected epoch 6 (0.918749).
The sequence variant uses MobileNetV3, 25 fitted-wire tokens, two transformer layers and
a 25-class absolute index at each token. It supervises labelled wires only and selects by
validation per-wire accuracy. Its GPU attempt produced no checkpoint in ten minutes; it
was restarted on CPU, four threads, FP32, batch 2, with eight epochs planned.
The concurrent v6 training jobs were never signalled.

**Limitation established before evaluating stage 2:** in the existing raw-832 reports,
most errors are mixed local slips. Without dropping predictions, frames with a uniform
GT index shift have macro fraction **0.531**, versus stage-1 numbering 0.522; with the old
wire gates this fraction is **0.563**, versus numbering 0.554. These are oracle feasibility
figures, not learned results. The failing 09-06 frame contains both 0 and −1 shifts.
A global offset alone therefore cannot meet the requested 0.601 / 1.000 goals while
retaining those predictions. We also test `--wire-reassign`: assign observed points to
nearby integer fret coordinates in the robust fit, retaining their pixel positions and
sides. Ambiguous points retain their original slot; collisions keep the higher-confidence
observation. Geometry-only is reported as a control.

Stage-2 diagnostics are pooled, separate from the macro headline: GT-strip offset accuracy
(target zero) and stage-1-strip accuracy against the dominant GT correction, with the
always-zero baseline and uniform-offset feasibility alongside them. High dominant-offset
accuracy does not imply every wire is correctly numbered. Reports also contain a balanced
GT-strip test across all seven offsets; sequence reports additionally score individual
labelled wires. The always-zero baseline on predicted strips is 0.940, so synthetic-offset
accuracy alone is insufficient evidence for a useful correction stage.

| Variant | eval v2 D13 mean / median px | Numbering | Coverage / detection | 09-06 D13 px | 09-06 numbering | 09-06 coverage | GT / stage-1 offset accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Geometry-only control](v2/v7_geometry_control.md) | 5.56 / 2.24 | 0.620 | 0.879 / 1.000 | 0.78 | 0.933 | 0.990 | — |
| [Geometry + reject ambiguous](v2/v7_geometry_reject.md) | 5.37 / 2.21 | 0.665 | 0.870 / 1.000 | 0.77 | 1.000 | 0.989 | — |
| [R18 1024 offset](v2/v7num_1024.md) | 11.74 / 6.30 | 0.450 | 0.909 / 1.000 | 0.78 | 0.933 | 0.990 | 0.901 / 0.793 |
| [R18 1024 offset + geometry/reject](v2/v7num_1024_reject.md) | 10.14 / 5.82 | 0.582 | 0.865 / 1.000 | 0.77 | 1.000 | 0.989 | 0.901 / 0.793 |
| [R18 1024 source pixels + geometry/reject](v2/v7num_1024_source_reject.md) | 10.00 / 5.76 | 0.576 | 0.864 / 1.000 | 0.77 | 1.000 | 0.989 | 0.901 / 0.780 |
| [MobileNet 1536 offset](v2/v7num_1536.md) | 8.44 / 2.74 | 0.522 | 0.910 / 1.000 | 0.78 | 0.933 | 0.990 | 0.884 / 0.797 |

<!-- two-stage-results-end -->

The completed geometry/reject control meets the brief's numerical targets: numbering
0.665 exceeds v1's 0.601, D13 5.37 / 2.21 px improves on 8.44 / 5.80, and 09-06 is
0.769 px / 1.000 numbering / 0.989 coverage. Coverage remains above v1 on both sets.
`--wire-reject-ambiguous` drops points at least 0.45 fret from the nearest fitted wire.
This rejection policy was introduced after inspecting the first control's one failing
09-06 frame, so this is an evaluation-informed development result. No held-out frame was
used for training or checkpoint selection.

The completed learned offset classifiers do not improve the geometry control. ResNet
scores 0.925 on balanced GT offsets but 0.793 on predicted strips, and makes 26 confident
nonzero corrections; MobileNet scores 0.860 / 0.797 and makes 17. Using original image
pixels with the **identical stage-1 fit** changes the ResNet predicted-strip score to
0.780 and combined numbering to 0.576, so restoring image detail alone does not resolve
the gap. These results support testing real stage-1 warps and natural correction frequencies
in future training, with geometry-only retained as the comparison.

Commands and artifact provenance: [reproduction](two_stage_reproduction.md),
[environment and data hashes](v2/v7_environment.json),
[ResNet training log](v2/v7num_1024_training.json),
[MobileNet training log](v2/v7num_1536_training.json).
At the last status check, retrieval of the MobileNet+geometry evaluation and sequence
CPU run was pending because PC SSH became unreachable. No result is claimed for either
until its report is recovered.
