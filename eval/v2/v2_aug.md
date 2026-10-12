# Eval v2: v2_aug

Weights `runs/pose/v2_aug/weights/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 6.27 / 4.42 | 17.5 | 5.78 | 0.692 | 0.960 | 0.997 |
| eval v2 pooled | 232 | 5.84 / 1.77 | 16.5 | 5.76 | 0.776 | 0.976 | 0.996 |
| eval v2 hand-corrected frames only | 1 | 1.94 / 1.69 | 4.9 | 1.94 | 1.000 | 0.974 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.04 / 0.81 | 4.3 | 1.04 | 1.000 | 0.915 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 6.93 / 5.68 | 28.3 | 7.76 | 0.000 | 0.997 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 0.87 / 0.73 | 5.5 | 1.29 | 0.958 | 0.993 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 33.62 / 28.36 | 61.6 | 18.14 | 0.286 | 0.970 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.94 / 1.69 | 4.9 | 1.94 | 1.000 | 0.974 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 2.40 / 1.71 | 13.7 | 3.46 | 0.231 | 0.789 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 2.37 / 1.71 | 9.2 | 7.46 | 0.958 | 0.987 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 5.32 / 3.37 | 30.0 | 9.47 | 0.375 | 0.971 | 0.958 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.25 / 1.81 | 6.9 | 4.40 | 0.958 | 0.998 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 2.29 / 1.96 | 7.7 | 3.17 | 1.000 | 0.990 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.63 / 1.42 | 3.8 | 2.15 | 1.000 | 0.939 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.55 / 2.25 | 6.0 | 6.51 | 1.000 | 0.993 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 9.86 / 8.52 | 23.6 | 6.75 | 0.000 | 0.850 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.74 / 1.24 | 37.5 | 1.74 | 0.917 | 0.996 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.04 / 1.49 | 5.7 | 6.65 | 1.000 | 0.993 | 1.000 |
