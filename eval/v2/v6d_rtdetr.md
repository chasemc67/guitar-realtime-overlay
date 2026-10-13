# Eval v2: v6d_rtdetr

Weights `runs/fretnet/v6d_rtdetr/best.pt`, imgsz 640. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 11.14 / 8.76 | 57.1 | 3.72 | 0.556 | 0.911 | 1.000 |
| eval v2 pooled | 232 | 9.75 / 1.68 | 45.5 | 4.05 | 0.470 | 0.908 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 1.43 / 1.11 | 3.6 | 1.43 | 1.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 1.02 / 0.76 | 4.2 | 1.02 | 0.933 | 0.993 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 13.96 / 11.80 | 56.7 | 6.52 | 0.000 | 0.992 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 90.39 / 88.33 | 569.7 | 1.68 | 0.000 | 0.515 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 6.73 / 1.23 | 10.2 | 4.99 | 0.786 | 0.968 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.43 / 1.11 | 3.6 | 1.43 | 1.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 2.74 / 0.83 | 15.5 | 2.37 | 0.000 | 0.890 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 2.41 / 1.63 | 9.4 | 6.41 | 0.250 | 0.977 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 8.74 / 4.65 | 50.0 | 6.56 | 0.042 | 0.582 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 3.81 / 2.83 | 11.6 | 4.13 | 0.625 | 0.967 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 5.33 / 4.62 | 17.9 | 4.97 | 0.250 | 0.969 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 1.00 / 0.75 | 2.3 | 1.83 | 1.000 | 1.000 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 1.86 / 1.36 | 4.4 | 4.46 | 1.000 | 0.988 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 1.87 / 1.44 | 4.5 | 1.87 | 1.000 | 0.950 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.20 / 0.62 | 36.2 | 0.97 | 0.917 | 0.996 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 2.42 / 1.49 | 6.8 | 3.90 | 0.917 | 0.960 | 1.000 |
