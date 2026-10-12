# Eval v2: v4q_r50_832_wgate

Weights `runs/fretnet/v4q_r50_832/best_wgate.pt`, imgsz 832. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 5.64 / 2.05 | 24.4 | 3.39 | 0.554 | 0.854 | 1.000 |
| eval v2 pooled | 232 | 5.34 / 1.11 | 20.2 | 3.37 | 0.461 | 0.854 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 1.76 / 1.72 | 4.5 | 1.76 | 1.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 0.78 / 0.51 | 3.2 | 0.78 | 0.933 | 0.990 | 1.000 |

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 8.53 / 1.90 | 34.7 | 6.57 | 0.000 | 0.818 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 4.44 / 0.73 | 27.8 | 0.99 | 0.042 | 0.704 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 1.86 / 0.85 | 3.1 | 5.29 | 0.929 | 0.942 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.76 / 1.72 | 4.5 | 1.76 | 1.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 2.89 / 0.83 | 16.6 | 2.30 | 0.000 | 0.675 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.59 / 1.74 | 17.9 | 5.80 | 0.042 | 0.852 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 29.46 / 11.31 | 165.5 | 7.96 | 0.000 | 0.496 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.40 / 1.34 | 7.4 | 2.91 | 0.542 | 0.864 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 3.27 / 2.04 | 11.0 | 3.33 | 0.292 | 0.872 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 0.92 / 0.71 | 2.1 | 1.67 | 1.000 | 0.992 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 1.82 / 1.40 | 4.3 | 2.78 | 1.000 | 0.946 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 2.20 / 2.57 | 5.2 | 2.20 | 1.000 | 0.850 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.29 / 0.58 | 36.4 | 0.94 | 0.917 | 0.987 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.59 / 1.03 | 4.5 | 3.01 | 1.000 | 0.951 | 1.000 |
