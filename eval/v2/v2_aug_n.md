# Eval v2: v2_aug_n

Weights `runs/pose/v2_aug_n/weights/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 6.18 / 2.82 | 18.0 | 5.38 | 0.759 | 0.957 | 0.994 |
| eval v2 pooled | 232 | 5.75 / 1.63 | 16.9 | 5.25 | 0.763 | 0.974 | 0.991 |
| eval v2 hand-corrected frames only | 1 | 1.40 / 1.14 | 3.5 | 1.40 | 1.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.08 / 0.84 | 4.4 | 1.07 | 0.933 | 0.915 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 8.26 / 7.22 | 33.6 | 8.04 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 0.80 / 0.64 | 5.0 | 0.81 | 1.000 | 0.972 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 29.36 / 5.18 | 48.7 | 12.16 | 0.714 | 0.980 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.40 / 1.14 | 3.5 | 1.40 | 1.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 3.82 / 2.47 | 21.9 | 3.52 | 0.077 | 0.816 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 2.86 / 2.19 | 11.2 | 7.78 | 0.708 | 0.989 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 7.35 / 3.72 | 42.1 | 9.00 | 0.292 | 0.937 | 0.917 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.58 / 2.01 | 7.8 | 4.55 | 0.917 | 0.991 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 1.27 / 1.05 | 4.3 | 2.20 | 1.000 | 0.990 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.68 / 1.44 | 3.9 | 2.19 | 1.000 | 0.939 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.00 / 1.65 | 4.7 | 5.94 | 1.000 | 0.993 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 9.35 / 7.86 | 22.3 | 9.35 | 1.000 | 0.800 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.38 / 0.78 | 36.6 | 1.41 | 0.917 | 0.996 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.37 / 2.05 | 6.7 | 6.96 | 1.000 | 0.993 | 1.000 |
