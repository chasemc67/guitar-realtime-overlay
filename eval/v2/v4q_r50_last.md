# Eval v2: v4q_r50_last

Weights `runs/fretnet/v4q_r50/last.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 6.73 / 2.79 | 26.7 | 4.05 | 0.441 | 0.951 | 1.000 |
| eval v2 pooled | 232 | 6.34 / 1.73 | 25.4 | 4.26 | 0.405 | 0.947 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 15.17 / 1.67 | 38.4 | 2.33 | 0.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.12 / 0.79 | 4.6 | 1.11 | 0.933 | 0.973 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 13.22 / 9.51 | 53.6 | 7.20 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 9.60 / 1.92 | 60.6 | 2.03 | 0.000 | 0.974 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 2.16 / 0.98 | 3.8 | 5.31 | 0.714 | 0.966 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 15.17 / 1.67 | 38.4 | 2.33 | 0.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 3.68 / 1.06 | 21.1 | 2.36 | 0.000 | 0.842 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 5.49 / 2.40 | 21.5 | 6.25 | 0.000 | 0.913 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 16.56 / 9.43 | 94.6 | 9.40 | 0.000 | 0.853 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.29 / 2.12 | 10.0 | 4.15 | 0.292 | 0.968 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 4.55 / 3.68 | 15.4 | 4.59 | 0.292 | 0.913 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.05 / 0.75 | 2.4 | 2.18 | 1.000 | 0.992 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.16 / 1.79 | 5.1 | 4.86 | 1.000 | 0.988 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 1.82 / 1.58 | 4.4 | 1.82 | 1.000 | 0.950 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.26 / 0.59 | 36.3 | 0.98 | 0.917 | 0.989 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.15 / 1.50 | 6.1 | 3.20 | 0.958 | 0.959 | 1.000 |
