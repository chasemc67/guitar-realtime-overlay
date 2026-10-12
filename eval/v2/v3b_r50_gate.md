# Eval v2: v3b_r50_gate

Weights `runs/fretnet/v3b_r50/best_gate.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 5.56 / 2.86 | 20.4 | 4.01 | 0.389 | 0.885 | 1.000 |
| eval v2 pooled | 232 | 5.30 / 1.78 | 18.4 | 4.03 | 0.424 | 0.901 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 1.53 / 1.02 | 3.9 | 1.66 | 0.000 | 0.921 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.11 / 0.77 | 4.5 | 1.10 | 0.867 | 0.990 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 11.38 / 8.01 | 46.2 | 7.27 | 0.000 | 0.983 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 1.82 / 1.04 | 11.5 | 1.47 | 0.000 | 0.880 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 1.97 / 1.15 | 3.5 | 5.24 | 0.857 | 0.919 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.53 / 1.02 | 3.9 | 1.66 | 0.000 | 0.921 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 4.73 / 1.45 | 26.6 | 2.56 | 0.000 | 0.764 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.31 / 2.48 | 16.8 | 5.65 | 0.042 | 0.892 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 12.49 / 10.82 | 70.3 | 9.95 | 0.000 | 0.441 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.70 / 2.44 | 11.2 | 4.04 | 0.500 | 0.960 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 5.17 / 4.46 | 17.4 | 5.27 | 0.208 | 0.944 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.33 / 0.82 | 3.1 | 2.13 | 1.000 | 0.977 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.85 / 2.25 | 6.7 | 4.45 | 1.000 | 0.945 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 11.21 / 1.97 | 26.8 | 1.99 | 0.000 | 0.825 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.27 / 0.61 | 36.4 | 0.83 | 0.917 | 0.990 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.03 / 1.54 | 5.7 | 3.57 | 0.917 | 0.951 | 1.000 |
