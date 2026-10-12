# Model v2 candidates vs v1 (eval set v2 + 09-06 honest set)

Every candidate is scored by `scripts/eval_v2.py` on the frozen eval set v2 (`configs/eval_v2.json`, 14 held-out
non-Ibanez clips, 232 frames) and the 09-06 hand-labelled honest set (15 frames). Full tables: `eval/v2/<name>.md`.

**Promotion rule:** a candidate replaces v1 only if eval-v2 macro D13 (along-neck px, the headline) improves without
regressing numbering or coverage (macro, and on the 09-06 honest set).

## Candidates

| Name | Base | Data | Augmentation | Epochs | Status |
|---|---|---|---|---:|---|
| v1 | yolo11s-pose, 640 | scale-v1 (3883 train frames) | degrees 5, scale 0.2, translate 0.1, persp 0.0005, hsv_s 0.3, hsv_v 0.2 | 150 | baseline |
| v2_aug | yolo11s-pose, 640 | scale-v1aug: scale-v1 + 758 synthetic fret-wrap + 1157 synthetic finger-occlusion copies (5798 train) | degrees 15, scale 0.5, translate 0.15, persp 0.001, hsv_s 0.5, hsv_v 0.35, erasing 0.4 | 150 | training |

## Headline (macro over clips)

| Model | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection | 09-06 D13 px | 09-06 numbering | 09-06 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 8.44 / 5.80 | 26.2 | 5.91 | 0.601 | 0.805 | 0.835 | 0.82 | 1.000 | 0.949 |
| v2_aug | TBD | | | | | | | | |

## Per guitar (D13 mean px / numbering / coverage)

| Clip (guitar) | Frames | v1 | v2_aug |
|---|---:|---|---|
| yt_9oJLRxucpNI (Gretsch 6120, hump-block) | 11 | 7.04 / 0.000 / 1.000 | |
| yt_FbEe_5tGOFM (Gibson SG Special, dots, 480p) | 24 | 1.05 / 1.000 / 0.981 | |
| yt_cgYA0Xt417w (Gibson SG '61, trapezoid, steep close-up) | 14 | 44.71 / 0.143 / 0.766 | |
| yt_GEljpsfLj8o (Les Paul goldtop, dark) | 1 | 2.22 / 1.000 / 0.947 | |
| yt_N_nayTl9HGY (Rickenbacker 360, triangles, 480p) | 13 | 3.78 / 0.077 / 0.567 | |
| yt_ixOqhZfAjMg (PRS singlecut, birds, wide) | 24 | 2.38 / 0.792 / 0.984 | |
| yt_j4D7XFUK7II (Jackson V, shark fins, 480p steep) | 24 | 18.01 / 0.000 / 0.168 | |
| yt_eM8DB4FT-jM (ESP George Lynch, dots) | 24 | 2.87 / 0.750 / 0.986 | |
| yt_yIYrqh7ulSk (Fender Jazzmaster, blocks) | 24 | 3.87 / 0.875 / 0.959 | |
| yt_VT4VkOkFIRE (ESP Eclipse-style, flags) | 3 | 3.64 / 1.000 / 0.939 | |
| yt_nBpe2YQEzZo (Fender Strat, maple) | 21 | 3.92 / 0.857 / 0.980 | |
| yt_tNAX9maMla8 (Superstrat, no face inlays) | 1 | n/a / 0.000 / 0.000 | |
| yt_vXzydcRC1bA (ESP superstrat, dots) | 24 | 13.69 / 0.917 / 0.996 | |
| stan_june2 (Telecaster, maple, phone) | 24 | 2.54 / 1.000 / 0.991 | |

v1 failure modes on eval v2: numbering on unfamiliar inlays (Gretsch, Rickenbacker, Jackson shark fins) and steep
close-ups where the headstock is out of view (SG '61, Jackson V: low detection/coverage).

## Latency

TBD (CoreML on the Mini, `scripts/bench_coreml.py`; v1 numbers in `eval/coreml_v1.md`).

## Decision

TBD.
