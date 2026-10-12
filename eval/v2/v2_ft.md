# Eval v2: v2_ft

Weights `runs/pose/v2_ft/weights/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 5.21 / 3.64 | 18.1 | 5.50 | 0.692 | 0.865 | 0.913 |
| eval v2 pooled | 232 | 4.89 / 2.10 | 15.3 | 5.47 | 0.685 | 0.903 | 0.888 |
| eval v2 hand-corrected frames only | 1 | 3.04 / 2.27 | 7.7 | 3.04 | 1.000 | 0.947 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.11 / 0.89 | 4.5 | 1.11 | 1.000 | 0.957 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 9.05 / 8.44 | 36.8 | 7.77 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 0.86 / 0.68 | 5.4 | 0.86 | 1.000 | 0.938 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 8.06 / 6.03 | 14.7 | 13.40 | 0.357 | 0.794 | 0.857 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 3.04 / 2.27 | 7.7 | 3.04 | 1.000 | 0.947 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 4.35 / 3.15 | 24.9 | 3.88 | 0.000 | 0.702 | 0.846 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 3.41 / 2.82 | 13.3 | 8.36 | 0.833 | 0.982 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 9.95 / 7.98 | 58.7 | 5.91 | 0.000 | 0.099 | 0.083 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.19 / 2.93 | 9.6 | 5.40 | 0.958 | 0.998 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 2.05 / 1.30 | 6.8 | 2.85 | 1.000 | 0.983 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 2.34 / 2.08 | 5.5 | 2.82 | 1.000 | 0.939 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.38 / 2.09 | 5.6 | 6.24 | 1.000 | 0.993 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 6.53 / 6.84 | 15.6 | 6.53 | 1.000 | 0.750 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.93 / 1.25 | 38.0 | 1.79 | 0.917 | 0.994 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 3.78 / 3.12 | 10.6 | 8.19 | 0.625 | 0.993 | 1.000 |
