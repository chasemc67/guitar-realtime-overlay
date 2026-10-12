# Eval v2: v4r_r50_last_s1000

Weights `runs/fretnet/v4r_r50/last_s1000.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 7.72 / 2.98 | 33.1 | 4.09 | 0.429 | 0.942 | 1.000 |
| eval v2 pooled | 232 | 7.66 / 1.73 | 33.0 | 4.30 | 0.388 | 0.942 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 11.85 / 1.77 | 30.0 | 3.05 | 0.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.07 / 0.77 | 4.3 | 1.05 | 0.867 | 0.984 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 12.96 / 8.11 | 52.6 | 6.95 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 20.02 / 2.36 | 126.2 | 2.10 | 0.000 | 0.970 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 2.52 / 0.94 | 4.3 | 4.78 | 0.714 | 0.962 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 11.85 / 1.77 | 30.0 | 3.05 | 0.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 3.99 / 0.79 | 23.0 | 2.38 | 0.000 | 0.713 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.73 / 2.16 | 18.5 | 6.05 | 0.000 | 0.923 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 21.90 / 12.03 | 124.0 | 9.49 | 0.000 | 0.809 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.69 / 2.22 | 11.3 | 4.52 | 0.250 | 0.969 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 5.83 / 4.88 | 19.7 | 5.20 | 0.125 | 0.952 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.08 / 0.77 | 2.5 | 1.89 | 1.000 | 1.000 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 2.33 / 1.96 | 5.5 | 4.33 | 1.000 | 0.986 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 1.97 / 1.75 | 4.7 | 1.97 | 1.000 | 0.950 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.23 / 0.59 | 36.2 | 0.94 | 0.917 | 0.993 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.95 / 1.45 | 5.5 | 3.59 | 1.000 | 0.962 | 1.000 |
