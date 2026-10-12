# Eval v2: v3c_r50nb_last

Weights `runs/fretnet/v3c_r50nb/last.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 7.53 / 2.62 | 26.3 | 3.93 | 0.406 | 0.914 | 1.000 |
| eval v2 pooled | 232 | 5.53 / 1.55 | 20.4 | 3.89 | 0.457 | 0.912 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 11.05 / 2.06 | 28.0 | 2.24 | 0.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.78 / 1.06 | 7.3 | 1.30 | 0.800 | 0.986 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 11.94 / 8.54 | 48.5 | 6.37 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 3.32 / 0.84 | 20.8 | 1.44 | 0.000 | 0.919 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 3.00 / 1.36 | 5.4 | 5.04 | 0.643 | 0.958 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 11.05 / 2.06 | 28.0 | 2.24 | 0.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 4.52 / 1.32 | 25.7 | 2.68 | 0.000 | 0.854 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 3.88 / 2.05 | 15.2 | 5.44 | 0.167 | 0.847 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 16.17 / 7.90 | 91.3 | 9.69 | 0.000 | 0.710 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.15 / 2.42 | 9.6 | 3.98 | 0.333 | 0.955 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 3.33 / 2.56 | 11.2 | 3.49 | 0.667 | 0.933 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.06 / 0.75 | 2.5 | 2.01 | 1.000 | 1.000 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.25 / 1.93 | 5.3 | 4.44 | 1.000 | 0.886 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 25.88 / 2.79 | 61.8 | 3.59 | 0.000 | 0.825 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.96 / 0.62 | 38.3 | 0.73 | 0.917 | 0.939 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.87 / 1.46 | 5.3 | 3.82 | 0.958 | 0.963 | 1.000 |
