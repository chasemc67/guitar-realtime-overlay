# Eval v2: v7num_1024_source_reject

Weights `runs/fretnet/v4q_r50_832/last.pt`, imgsz 832. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 10.00 / 5.76 | 39.3 | 3.47 | 0.576 | 0.864 | 1.000 |
| eval v2 pooled | 232 | 6.73 / 1.22 | 26.8 | 3.41 | 0.563 | 0.870 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 44.03 / 37.75 | 111.6 | 1.76 | 0.000 | 0.947 | 1.000 |
| 09-06 hand GT (honest) | 15 | 0.77 / 0.51 | 3.1 | 0.77 | 1.000 | 0.989 | 1.000 |

Stage-2 checkpoint: `runs/fretnet/v7num_1024/best.pt`, confidence threshold 0.8.
Classifier diagnostics (pooled; GT strips have target offset zero):

```json
{
  "eval_v2": {
    "frames": 232,
    "gt_fit_frames": 232,
    "stage1_fit_scored_frames": 232,
    "gt_strip_offset_accuracy": 0.9008620689655172,
    "gt_strip_balanced_offset_accuracy": 0.9254926108374385,
    "stage1_strip_offset_accuracy": 0.7801724137931034,
    "zero_offset_baseline_accuracy": 0.9396551724137931,
    "uniform_offset_possible_fraction": 0.4267241379310345,
    "nonzero_changes": 24
  },
  "honest_0906": {
    "frames": 15,
    "gt_fit_frames": 15,
    "stage1_fit_scored_frames": 15,
    "gt_strip_offset_accuracy": 1.0,
    "gt_strip_balanced_offset_accuracy": 1.0,
    "stage1_strip_offset_accuracy": 1.0,
    "zero_offset_baseline_accuracy": 1.0,
    "uniform_offset_possible_fraction": 0.9333333333333333,
    "nonzero_changes": 0
  }
}
```

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 10.52 / 2.17 | 43.8 | 6.53 | 0.182 | 0.878 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 7.51 / 0.89 | 47.4 | 1.06 | 0.375 | 0.776 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 1.32 / 0.86 | 2.4 | 5.26 | 1.000 | 0.948 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 44.03 / 37.75 | 111.6 | 1.76 | 0.000 | 0.947 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 8.63 / 1.17 | 48.2 | 2.41 | 0.385 | 0.726 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 7.60 / 2.69 | 30.0 | 5.92 | 0.125 | 0.868 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 33.01 / 25.09 | 189.8 | 8.77 | 0.000 | 0.456 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.25 / 1.40 | 7.0 | 2.87 | 0.792 | 0.900 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 5.31 / 2.35 | 17.7 | 3.33 | 0.292 | 0.876 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 0.92 / 0.71 | 2.1 | 1.67 | 1.000 | 0.992 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 1.82 / 1.41 | 4.3 | 2.78 | 1.000 | 0.946 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 2.20 / 2.57 | 5.2 | 2.20 | 1.000 | 0.850 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.29 / 0.58 | 36.4 | 0.94 | 0.917 | 0.987 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.59 / 1.03 | 4.5 | 3.01 | 1.000 | 0.951 | 1.000 |
