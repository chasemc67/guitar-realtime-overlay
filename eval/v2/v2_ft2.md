# Eval v2: v2_ft2

Weights `runs/pose/v2_ft2/weights/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 9.12 / 7.60 | 23.3 | 5.76 | 0.758 | 0.954 | 0.989 |
| eval v2 pooled | 232 | 8.47 / 2.00 | 21.9 | 5.79 | 0.767 | 0.968 | 0.987 |
| eval v2 hand-corrected frames only | 1 | 1.75 / 1.28 | 4.4 | 1.75 | 1.000 | 0.947 | 1.000 |
| 09-06 hand GT (honest) | 15 | 0.89 / 0.61 | 3.6 | 0.89 | 0.933 | 0.915 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 10.16 / 9.31 | 41.4 | 8.07 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 1.36 / 1.11 | 8.5 | 1.92 | 1.000 | 0.996 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 73.05 / 70.95 | 130.9 | 18.26 | 0.071 | 0.964 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.75 / 1.28 | 4.4 | 1.75 | 1.000 | 0.947 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 2.49 / 1.48 | 14.1 | 3.67 | 0.538 | 0.709 | 0.923 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 2.63 / 2.04 | 10.3 | 7.56 | 0.833 | 0.984 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 6.10 / 4.39 | 35.1 | 9.26 | 0.250 | 0.954 | 0.917 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.24 / 2.22 | 6.8 | 4.35 | 1.000 | 0.995 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 2.08 / 1.86 | 7.0 | 2.88 | 1.000 | 0.982 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.48 / 1.23 | 3.5 | 1.99 | 1.000 | 0.939 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.36 / 2.29 | 5.5 | 6.31 | 1.000 | 0.993 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 5.74 / 5.20 | 13.7 | 5.74 | 1.000 | 0.900 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.78 / 1.17 | 37.6 | 1.67 | 0.917 | 0.993 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.41 / 1.81 | 6.8 | 7.15 | 1.000 | 0.993 | 1.000 |
