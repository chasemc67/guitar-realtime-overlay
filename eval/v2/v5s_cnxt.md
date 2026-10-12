# Eval v2: v5s_cnxt

Weights `runs/fretnet/v5s_cnxt/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 18.14 / 16.41 | 66.8 | 9.39 | 0.207 | 0.946 | 1.000 |
| eval v2 pooled | 232 | 16.49 / 10.31 | 62.4 | 9.03 | 0.208 | 0.950 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 27.21 / 20.09 | 68.9 | 14.19 | 0.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 3.80 / 3.66 | 15.5 | 3.77 | 0.533 | 0.987 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 26.96 / 31.08 | 109.1 | 8.88 | 0.000 | 1.000 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 40.89 / 39.77 | 257.7 | 4.69 | 0.000 | 0.854 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 45.18 / 46.21 | 80.2 | 15.46 | 0.000 | 0.946 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 27.21 / 20.09 | 68.9 | 14.19 | 0.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 5.57 / 4.93 | 31.7 | 5.79 | 0.000 | 0.882 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 5.40 / 4.62 | 21.1 | 9.63 | 0.542 | 0.966 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 28.36 / 23.73 | 160.9 | 9.42 | 0.000 | 0.800 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 9.59 / 8.50 | 28.7 | 10.43 | 0.042 | 0.997 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 11.54 / 11.27 | 39.1 | 9.18 | 0.042 | 0.982 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 4.99 / 4.42 | 11.6 | 5.33 | 1.000 | 1.000 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 8.92 / 8.74 | 21.0 | 11.63 | 0.190 | 0.978 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 10.89 / 9.92 | 26.0 | 10.50 | 0.000 | 0.900 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 14.29 / 1.66 | 38.9 | 2.14 | 0.917 | 0.995 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 14.17 / 14.75 | 40.2 | 14.16 | 0.167 | 0.950 | 1.000 |
