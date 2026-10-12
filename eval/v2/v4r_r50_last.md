# Eval v2: v4r_r50_last

Weights `runs/fretnet/v4r_r50/last.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 9.91 / 6.51 | 44.2 | 4.84 | 0.431 | 0.942 | 1.000 |
| eval v2 pooled | 232 | 9.83 / 2.76 | 45.4 | 4.87 | 0.388 | 0.942 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 17.18 / 4.65 | 43.5 | 5.94 | 0.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.17 / 0.91 | 4.8 | 1.16 | 0.933 | 0.984 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 13.51 / 11.62 | 54.8 | 7.03 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 39.62 / 38.02 | 249.9 | 2.99 | 0.000 | 0.970 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 3.66 / 1.97 | 5.9 | 6.02 | 0.786 | 0.962 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 17.18 / 4.65 | 43.5 | 5.94 | 0.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 3.42 / 1.37 | 19.7 | 2.69 | 0.000 | 0.713 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.88 / 3.30 | 19.0 | 6.88 | 0.000 | 0.923 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 23.27 / 12.27 | 131.3 | 10.24 | 0.000 | 0.809 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 4.08 / 3.49 | 12.4 | 5.13 | 0.167 | 0.969 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 6.31 / 5.91 | 21.4 | 5.87 | 0.167 | 0.952 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.02 / 0.85 | 2.4 | 1.98 | 1.000 | 1.000 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.13 / 1.68 | 5.0 | 4.10 | 1.000 | 0.986 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 3.45 / 2.91 | 8.2 | 3.45 | 1.000 | 0.950 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.27 / 0.64 | 36.4 | 0.96 | 0.917 | 0.993 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.95 / 2.41 | 8.4 | 4.52 | 1.000 | 0.962 | 1.000 |
