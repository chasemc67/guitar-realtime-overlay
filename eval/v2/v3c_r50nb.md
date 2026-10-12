# Eval v2: v3c_r50nb

Weights `runs/fretnet/v3c_r50nb/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 6.61 / 2.48 | 25.7 | 3.66 | 0.377 | 0.894 | 1.000 |
| eval v2 pooled | 232 | 6.21 / 1.63 | 23.9 | 3.58 | 0.427 | 0.893 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 6.23 / 1.70 | 15.8 | 1.94 | 0.000 | 0.974 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.54 / 0.99 | 6.3 | 1.23 | 0.733 | 0.982 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 11.80 / 7.93 | 47.9 | 6.20 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 9.91 / 1.02 | 62.2 | 1.34 | 0.000 | 0.853 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 5.08 / 1.55 | 8.9 | 4.38 | 0.357 | 0.944 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 6.23 / 1.70 | 15.8 | 1.94 | 0.000 | 0.974 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 2.98 / 1.20 | 17.0 | 2.51 | 0.000 | 0.846 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.28 / 2.00 | 16.8 | 5.05 | 0.292 | 0.824 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 16.64 / 7.30 | 94.2 | 9.55 | 0.000 | 0.666 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.37 / 2.62 | 10.2 | 3.81 | 0.250 | 0.946 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 4.04 / 2.73 | 13.7 | 3.40 | 0.500 | 0.898 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.08 / 0.78 | 2.5 | 2.04 | 1.000 | 0.992 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.29 / 1.91 | 5.4 | 3.85 | 1.000 | 0.901 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 8.77 / 1.85 | 20.9 | 3.32 | 0.000 | 0.800 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 14.12 / 0.62 | 38.7 | 0.74 | 0.917 | 0.927 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.91 / 1.46 | 5.4 | 3.08 | 0.958 | 0.949 | 1.000 |
