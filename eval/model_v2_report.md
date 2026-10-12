# Model v2 candidates vs v1 (eval set v2 + 09-06 honest set), 2026-10-12

Every candidate is scored by `scripts/eval_v2.py` on the frozen eval set v2 (`configs/eval_v2.json`, 14 held-out
non-Ibanez clips, 232 frames) and the 09-06 hand-labelled honest set (15 frames, Chase's 24-fret Ibanez). Full
tables: `eval/v2/<name>.md`, raw `eval/v2/<name>.json`.

**Promotion rule:** a candidate replaces v1 only if eval-v2 macro D13 (along-neck px, the headline) improves without
regressing numbering or coverage (eval-v2 macro and the 09-06 honest set).

**Result: no candidate passes. v1 stays the shipped model.** v2_aug is the strongest candidate and the base for the
next attempt (see Decision).

## Candidates

yolo11s-pose unless noted, imgsz 640, trained on the 4090 (`scripts/train_yolo.py`; weights stay on the PC under
`~/gro/repo/runs/pose/<name>/`).

| Name | Init | Data | Augmentation | Schedule |
|---|---|---|---|---|
| v1 | COCO pose | scale-v1 (3883 train frames) | degrees 5, scale 0.2, translate 0.1, persp 0.0005, hsv_s 0.3, hsv_v 0.2 | 150 ep, auto optimizer |
| v2_aug | COCO pose | scale-v1aug: scale-v1 + 758 synthetic fret-wrap + 1157 synthetic finger-occlusion copies (5798 train; `export_yolo.py --wrap-aug/--occ-aug`) | degrees 15, scale 0.5, translate 0.15, persp 0.001, hsv_s 0.5, hsv_v 0.35, erasing 0.4 | 150 ep, auto optimizer |
| v2_ft | v2_aug best | scale-v1aug | v1's | 30 ep, auto optimizer (MuSGD lr 0.01: `optimizer=auto` ignores lr0) |
| v2_ft2 | v2_aug best | scale-v1aug | v1's | 15 ep, SGD lr 0.001, no warmup |
| v2_aug_n | COCO pose, **yolo11n-pose** | scale-v1aug | v2_aug's | 150 ep, auto optimizer |

No new guitars were added to training in this round: all v2 candidates see the same clips as v1.

## Headline

| Model | eval v2 D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection | 09-06 D13 px | 09-06 numbering | 09-06 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 8.44 / 5.80 | 26.2 | 5.91 | 0.601 | 0.805 | 0.835 | **0.82** | 1.000 | 0.949 |
| v2_aug | 6.27 / 4.42 | **17.5** | 5.78 | 0.692 | **0.960** | **0.997** | 1.04 | 1.000 | 0.915 |
| v2_ft | **5.21** / **3.64** | 18.1 | **5.50** | 0.692 | 0.865 | 0.913 | 1.11 | 1.000 | **0.957** |
| v2_ft2 | 9.12 / 7.60 | 23.3 | 5.76 | 0.758 | 0.954 | 0.989 | 0.89 | 0.933 | 0.915 |
| v2_aug_n (yolo11n) | 6.18 / **2.82** | 18.0 | 5.38 | **0.759** | 0.957 | 0.994 | 1.08 | 0.933 | 0.915 |

Eval-v2 columns are macro over clips. D13 is averaged over detected frames only, so a model that misses hard frames
can look better than one that finds them (v1 detects 21% of the Jackson V frames, v2_ft 8%, v2_aug 96%).

## Per guitar (D13 mean px / numbering / coverage)

| Clip (guitar) | Frames | v1 | v2_aug | v2_ft | v2_ft2 | v2_aug_n |
|---|---:|---|---|---|---|---|
| yt_9oJLRxucpNI (Gretsch 6120, hump-block) | 11 | 7.04 / 0.00 / 1.00 | 6.93 / 0.00 / 1.00 | 9.05 / 0.00 / 1.00 | 10.16 / 0.00 / 1.00 | 8.26 / 0.00 / 1.00 |
| yt_FbEe_5tGOFM (Gibson SG Special, dots, 480p) | 24 | 1.05 / 1.00 / 0.98 | 0.87 / 0.96 / 0.99 | 0.86 / 1.00 / 0.94 | 1.36 / 1.00 / 1.00 | 0.80 / 1.00 / 0.97 |
| yt_cgYA0Xt417w (Gibson SG '61, trapezoid, steep close-up) | 14 | 44.71 / 0.14 / 0.77 | 33.62 / 0.29 / 0.97 | 8.06 / 0.36 / 0.79 | 73.05 / 0.07 / 0.96 | 29.36 / 0.71 / 0.98 |
| yt_GEljpsfLj8o (Les Paul goldtop, dark) | 1 | 2.22 / 1.00 / 0.95 | 1.94 / 1.00 / 0.97 | 3.04 / 1.00 / 0.95 | 1.75 / 1.00 / 0.95 | 1.40 / 1.00 / 1.00 |
| yt_N_nayTl9HGY (Rickenbacker 360, triangles, 480p) | 13 | 3.78 / 0.08 / 0.57 | 2.40 / 0.23 / 0.79 | 4.35 / 0.00 / 0.70 | 2.49 / 0.54 / 0.71 | 3.82 / 0.08 / 0.82 |
| yt_ixOqhZfAjMg (PRS singlecut, birds, wide) | 24 | 2.38 / 0.79 / 0.98 | 2.37 / 0.96 / 0.99 | 3.41 / 0.83 / 0.98 | 2.63 / 0.83 / 0.98 | 2.86 / 0.71 / 0.99 |
| yt_j4D7XFUK7II (Jackson V, shark fins, 480p steep) | 24 | 18.01 / 0.00 / 0.17 | 5.32 / 0.38 / 0.97 | 9.95 / 0.00 / 0.10 | 6.10 / 0.25 / 0.95 | 7.35 / 0.29 / 0.94 |
| yt_eM8DB4FT-jM (ESP George Lynch, dots) | 24 | 2.87 / 0.75 / 0.99 | 2.25 / 0.96 / 1.00 | 3.19 / 0.96 / 1.00 | 2.24 / 1.00 / 0.99 | 2.58 / 0.92 / 0.99 |
| yt_yIYrqh7ulSk (Fender Jazzmaster, blocks) | 24 | 3.87 / 0.88 / 0.96 | 2.29 / 1.00 / 0.99 | 2.05 / 1.00 / 0.98 | 2.08 / 1.00 / 0.98 | 1.27 / 1.00 / 0.99 |
| yt_VT4VkOkFIRE (ESP Eclipse-style, flags) | 3 | 3.64 / 1.00 / 0.94 | 1.63 / 1.00 / 0.94 | 2.34 / 1.00 / 0.94 | 1.48 / 1.00 / 0.94 | 1.68 / 1.00 / 0.94 |
| yt_nBpe2YQEzZo (Fender Strat, maple) | 21 | 3.92 / 0.86 / 0.98 | 2.55 / 1.00 / 0.99 | 2.38 / 1.00 / 0.99 | 2.36 / 1.00 / 0.99 | 2.00 / 1.00 / 0.99 |
| yt_tNAX9maMla8 (Superstrat, no face inlays) | 1 | n/a / 0.00 / 0.00 | 9.86 / 0.00 / 0.85 | 6.53 / 1.00 / 0.75 | 5.74 / 1.00 / 0.90 | 9.35 / 1.00 / 0.80 |
| yt_vXzydcRC1bA (ESP superstrat, dots) | 24 | 13.69 / 0.92 / 1.00 | 13.74 / 0.92 / 1.00 | 13.93 / 0.92 / 0.99 | 13.78 / 0.92 / 0.99 | 13.38 / 0.92 / 1.00 |
| stan_june2 (Telecaster, maple, phone) | 24 | 2.54 / 1.00 / 0.99 | 2.04 / 1.00 / 0.99 | 3.78 / 0.62 / 0.99 | 2.41 / 1.00 / 0.99 | 2.37 / 1.00 / 0.99 |

## Reading

- **v2_aug** (synthetic wrap/occlusion + stronger geometric/colour augmentation) is better than v1 on 12 of 14 guitars
  for D13 and never worse on numbering except SG Special (0.96 vs 1.00, one frame). It fixes v1's two big failures:
  coverage on the Jackson V (0.17 → 0.97) and detection on steep close-ups (eval-v2 detection 0.835 → 0.997).
- **Its one regression is on 09-06, Chase's 24-fret Ibanez:** coverage 0.949 → 0.915 and D13 0.82 → 1.04 px
  (both well under one fret: 4.3% of the local fret spacing). The lost slots are fret 23. Mean keypoint
  confidence of fret 23 over the 15 frames is 0.54 for v1 (barely over the 0.5 anchor threshold) and 0.26 for v2_aug;
  fret 24 is below threshold for both (0.23 / 0.10). The stronger scale augmentation lowered confidence on the
  rarely seen top frets.
- **The 09-06 slip does not change the tracker output.** `FretboardTracker` (torch backend, `end_every=1`, 450 frames)
  settles at 24 frets on 09-06 with both v1 and v2_aug, and the homography fit extrapolates fret 23/24 from the
  other wires.
- **v2_ft** (30-epoch fine-tune with v1's augmentation) has the best eval-v2 headline (5.21 px) and recovers fret 23
  on 09-06, so it passes the letter of the rule, but it is not promoted. Most of its headline gain is one clip
  (SG '61 44.7 → 8.1 px, on fewer detected frames). It breaks numbering on Stan (1.00 → 0.62), loses most Jackson V
  detections (0.08) and is worse than v2_aug on D13 for 9 of 14 guitars. The likely cause is the lr 0.01 restart
  (`optimizer=auto` ignored lr0), which overwrote much of what v2_aug learned.
- **v2_ft2** (gentle SG lr 0.001 fine-tune) keeps v2_aug's coverage and has the best numbering (0.758). But it
  degrades the SG '61 close-up (73 px), regresses 09-06 numbering (0.933, one frame) and loses frets 23 and 24
  there. It fails.
- **v2_aug_n** (yolo11n-pose, v2_aug recipe) matches v2_aug on eval v2 with a much smaller network (6.18 px, best
  median 2.82 px and best numbering 0.759, coverage 0.957). It has the same 09-06 problem, worse: predicted frets stop
  at 22 in every frame (coverage 0.915), plus one numbering miss (0.933). It fails the rule too, but the small model
  is not the bottleneck on unfamiliar guitars.

## Decision

**Keep v1.** v2_aug is clearly better on the new guitars but regresses 09-06 coverage (fret 23 keypoint), and the
rule forbids a coverage regression on the honest set. The one candidate that technically passes (v2_ft) is worse than
v2_aug almost everywhere and breaks numbering on Stan, so promoting it on the macro number would be misleading.

Next step: use v2_aug as the base and target high-fret confidence. Options: add 24-fret training clips (most YouTube
guitars are 21/22 frets), oversample frames with frets 23/24 labelled, or cap `scale` augmentation at about 0.3 so the
top of the board is not routinely cropped or shrunk. Then re-run this table. Train both s and n on that recipe: n is as accurate as s here and about 1.8x faster.

## Latency

v1 CoreML (Mac Mini M4, fp16, ALL): model 4.25 ms median, end to end ≈ 87–91 fps (`eval/coreml_v1.md`). The model
call is under half of the ~10 ms per-frame time; post-processing is the rest. A smaller model can save at most about
4 ms per frame.

yolo11n-pose (`v2_aug_n`), CoreML fp16 (`imgsz=640, nms=False, half=True`), same Mini, same session, 200 calls,
median / p95 ms (`eval/v2/bench_coreml_v2_aug_n.json`):

| Model | ALL | CPU_AND_NE | CPU_AND_GPU | load (ALL) |
|---|---|---|---|---:|
| v1 fp16 (yolo11s) | 4.33 / 5.39 | 4.28 / 4.59 | 12.1 / 12.4 | 2.29 s |
| v2_aug_n fp16 (yolo11n) | **2.48 / 2.65** | 2.48 / 2.64 | 6.49 / 6.94 | 0.72 s |

yolo11n saves about 1.8 ms per frame on the ANE (~10.4 → ~8.6 ms end to end, estimated from the v1 stage split;
the eval videos are no longer on the Mini, so end-to-end was not re-run). That is worth having on older devices
(the GPU-only path halves, 12.1 → 6.5 ms), but it is not needed on the M4.
