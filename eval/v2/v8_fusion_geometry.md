# Eval v2: v8_fusion_geometry

Weights `runs/fretnet/v4q_r50_832/last.pt`, imgsz 832. Frozen set `configs/eval_v2.json`. Macro = mean over clips.

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **eval v2 macro** | 232 | 6.20 / 2.93 | 28.5 | 3.46 | 0.665 | 0.868 | 1.000 |
| eval v2 pooled | 232 | 5.62 / 1.14 | 22.7 | 3.41 | 0.584 | 0.870 | 1.000 |
| eval v2 hand-corrected frames only | 1 | 1.76 / 1.72 | 4.5 | 1.76 | 1.000 | 1.000 | 1.000 |
| 09-06 hand GT (honest) | 15 | 0.77 / 0.51 | 3.1 | 0.77 | 1.000 | 0.989 | 1.000 |

Landmark checkpoint: `runs/fretnet/v8_landmark_1024/best.pt`; threshold 0.8.
Per-landmark detection diagnostics (pooled, localization tolerance 0.5 fret):

```json
{
  "eval_v2": {
    "predicted_strip": {
      "12": {
        "frames": 232,
        "present": 212,
        "detected": 187,
        "localized": 181,
        "presence_accuracy": 0.9008620689655172,
        "precision": 0.9679144385026738,
        "recall": 0.8537735849056604,
        "detected_mean_error_fret": 0.15471178125753418
      },
      "5": {
        "frames": 232,
        "present": 147,
        "detected": 98,
        "localized": 88,
        "presence_accuracy": 0.7758620689655172,
        "precision": 0.8979591836734694,
        "recall": 0.5986394557823129,
        "detected_mean_error_fret": 0.2031081453231084
      },
      "7": {
        "frames": 232,
        "present": 180,
        "detected": 128,
        "localized": 119,
        "presence_accuracy": 0.8189655172413793,
        "precision": 0.9296875,
        "recall": 0.6611111111111111,
        "detected_mean_error_fret": 0.10038187624516966
      },
      "9": {
        "frames": 232,
        "present": 202,
        "detected": 187,
        "localized": 174,
        "presence_accuracy": 0.8577586206896551,
        "precision": 0.93048128342246,
        "recall": 0.8613861386138614,
        "detected_mean_error_fret": 0.08154911329393398
      },
      "3": {
        "frames": 232,
        "present": 131,
        "detected": 69,
        "localized": 52,
        "presence_accuracy": 0.7887931034482759,
        "precision": 0.7536231884057971,
        "recall": 0.3969465648854962,
        "detected_mean_error_fret": 0.2887199720667596
      },
      "15": {
        "frames": 232,
        "present": 193,
        "detected": 135,
        "localized": 124,
        "presence_accuracy": 0.8362068965517241,
        "precision": 0.9185185185185185,
        "recall": 0.6424870466321243,
        "detected_mean_error_fret": 0.10943877688085861
      },
      "0": {
        "frames": 232,
        "present": 20,
        "detected": 0,
        "localized": 0,
        "presence_accuracy": 0.9137931034482759,
        "precision": null,
        "recall": 0.0,
        "detected_mean_error_fret": null
      }
    },
    "gt_strip": {
      "12": {
        "frames": 232,
        "present": 212,
        "detected": 215,
        "localized": 200,
        "presence_accuracy": 0.9051724137931034,
        "precision": 0.9302325581395349,
        "recall": 0.9433962264150944,
        "detected_mean_error_fret": 0.1546876975124385
      },
      "5": {
        "frames": 232,
        "present": 147,
        "detected": 91,
        "localized": 82,
        "presence_accuracy": 0.8793103448275862,
        "precision": 0.9010989010989011,
        "recall": 0.5578231292517006,
        "detected_mean_error_fret": 0.11694770295498817
      },
      "7": {
        "frames": 232,
        "present": 182,
        "detected": 130,
        "localized": 117,
        "presence_accuracy": 0.8189655172413793,
        "precision": 0.9,
        "recall": 0.6428571428571429,
        "detected_mean_error_fret": 0.11164864878540301
      },
      "9": {
        "frames": 232,
        "present": 203,
        "detected": 212,
        "localized": 192,
        "presence_accuracy": 0.8836206896551724,
        "precision": 0.9056603773584906,
        "recall": 0.9458128078817734,
        "detected_mean_error_fret": 0.07273823354503771
      },
      "3": {
        "frames": 232,
        "present": 131,
        "detected": 52,
        "localized": 41,
        "presence_accuracy": 0.8146551724137931,
        "precision": 0.7884615384615384,
        "recall": 0.31297709923664124,
        "detected_mean_error_fret": 0.1406509079639244
      },
      "15": {
        "frames": 232,
        "present": 194,
        "detected": 172,
        "localized": 153,
        "presence_accuracy": 0.8620689655172413,
        "precision": 0.8895348837209303,
        "recall": 0.788659793814433,
        "detected_mean_error_fret": 0.06599075116724212
      },
      "0": {
        "frames": 232,
        "present": 20,
        "detected": 0,
        "localized": 0,
        "presence_accuracy": 0.9137931034482759,
        "precision": null,
        "recall": 0.0,
        "detected_mean_error_fret": null
      }
    },
    "fit_failures": 0,
    "nonzero_corrections": 10
  },
  "honest_0906": {
    "predicted_strip": {
      "12": {
        "frames": 15,
        "present": 15,
        "detected": 11,
        "localized": 11,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.7333333333333333,
        "detected_mean_error_fret": 0.16798207771559379
      },
      "5": {
        "frames": 15,
        "present": 15,
        "detected": 0,
        "localized": 0,
        "presence_accuracy": 1.0,
        "precision": null,
        "recall": 0.0,
        "detected_mean_error_fret": null
      },
      "7": {
        "frames": 15,
        "present": 15,
        "detected": 5,
        "localized": 5,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.3333333333333333,
        "detected_mean_error_fret": 0.039379784591297096
      },
      "9": {
        "frames": 15,
        "present": 15,
        "detected": 13,
        "localized": 13,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.8666666666666667,
        "detected_mean_error_fret": 0.0823581823355844
      },
      "3": {
        "frames": 15,
        "present": 15,
        "detected": 7,
        "localized": 7,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.4666666666666667,
        "detected_mean_error_fret": 0.20588597784828064
      },
      "15": {
        "frames": 15,
        "present": 15,
        "detected": 10,
        "localized": 10,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.6666666666666666,
        "detected_mean_error_fret": 0.025611513467599423
      },
      "0": {
        "frames": 15,
        "present": 0,
        "detected": 0,
        "localized": 0,
        "presence_accuracy": 1.0,
        "precision": null,
        "recall": null,
        "detected_mean_error_fret": null
      }
    },
    "gt_strip": {
      "12": {
        "frames": 15,
        "present": 15,
        "detected": 14,
        "localized": 14,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.9333333333333333,
        "detected_mean_error_fret": 0.13237399131502528
      },
      "5": {
        "frames": 15,
        "present": 15,
        "detected": 5,
        "localized": 5,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.3333333333333333,
        "detected_mean_error_fret": 0.06779871190701811
      },
      "7": {
        "frames": 15,
        "present": 15,
        "detected": 10,
        "localized": 10,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.6666666666666666,
        "detected_mean_error_fret": 0.024949990785273712
      },
      "9": {
        "frames": 15,
        "present": 15,
        "detected": 14,
        "localized": 14,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.9333333333333333,
        "detected_mean_error_fret": 0.02826272721160031
      },
      "3": {
        "frames": 15,
        "present": 15,
        "detected": 4,
        "localized": 4,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.26666666666666666,
        "detected_mean_error_fret": 0.2730811545701022
      },
      "15": {
        "frames": 15,
        "present": 15,
        "detected": 14,
        "localized": 14,
        "presence_accuracy": 1.0,
        "precision": 1.0,
        "recall": 0.9333333333333333,
        "detected_mean_error_fret": 0.014601929386315706
      },
      "0": {
        "frames": 15,
        "present": 0,
        "detected": 0,
        "localized": 0,
        "presence_accuracy": 1.0,
        "precision": null,
        "recall": null,
        "detected_mean_error_fret": null
      }
    },
    "fit_failures": 0,
    "nonzero_corrections": 0
  }
}
```

## Per clip

| Set / clip | Frames | D13 mean / median px | D13 % fret | Nearest-wire px | Numbering | Coverage | Detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| yt_9oJLRxucpNI (Gretsch 6120 orange hollow body, ebony-ish board, hump-block inlays) | 11 | 5.43 / 1.80 | 22.3 | 6.52 | 0.182 | 0.881 | 1.000 |
| yt_FbEe_5tGOFM (Gibson SG Special cherry, rosewood, dot inlays, 640x480 12.6 fps) | 24 | 9.62 / 0.89 | 60.3 | 1.05 | 0.375 | 0.769 | 1.000 |
| yt_cgYA0Xt417w (Gibson SG Standard '61 cherry, rosewood, trapezoid inlays, 1080p24, steep close-up) | 14 | 1.32 / 0.86 | 2.4 | 5.26 | 1.000 | 0.948 | 1.000 |
| yt_GEljpsfLj8o (Gibson Les Paul goldtop, rosewood, trapezoid inlays, dark scene) | 1 | 1.76 / 1.72 | 4.5 | 1.76 | 1.000 | 1.000 | 1.000 |
| yt_N_nayTl9HGY (Rickenbacker 360 mapleglo, bound red-stained board, triangle inlays, 480p) | 13 | 1.50 / 0.84 | 8.7 | 2.41 | 0.538 | 0.741 | 1.000 |
| yt_ixOqhZfAjMg (PRS singlecut teal flame, rosewood, bird inlays, wide shot) | 24 | 4.42 / 1.76 | 17.4 | 5.93 | 0.208 | 0.863 | 1.000 |
| yt_j4D7XFUK7II (Jackson V red, ebony, shark-fin inlays, 480p steep angle, headstock out of view) | 24 | 37.44 / 23.43 | 212.8 | 8.72 | 0.000 | 0.452 | 1.000 |
| yt_eM8DB4FT-jM (ESP George Lynch tiger graphic, rosewood, dot inlays, reverse headstock) | 24 | 2.25 / 1.40 | 7.0 | 2.87 | 0.792 | 0.900 | 1.000 |
| yt_yIYrqh7ulSk (Fender Jazzmaster sunburst, rosewood, block inlays, 1080p) | 24 | 3.29 / 2.04 | 11.1 | 3.33 | 0.292 | 0.872 | 1.000 |
| yt_VT4VkOkFIRE (ESP Eclipse-style singlecut black, ebony, flag inlays (licklibrary close-up)) | 3 | 0.92 / 0.71 | 2.1 | 1.67 | 1.000 | 0.992 | 1.000 |
| yt_nBpe2YQEzZo (Fender Stratocaster sunburst, maple board, dot inlays) | 21 | 1.82 / 1.40 | 4.3 | 2.78 | 1.000 | 0.946 | 1.000 |
| yt_tNAX9maMla8 (Superstrat flame top, ebony, no face inlays (steep close-ups)) | 1 | 2.20 / 2.57 | 5.2 | 2.20 | 1.000 | 0.850 | 1.000 |
| yt_vXzydcRC1bA (ESP superstrat black, rosewood, dot inlays (licklibrary)) | 24 | 13.29 / 0.58 | 36.4 | 0.94 | 0.917 | 0.987 | 1.000 |
| stan_june2 (Fender Telecaster butterscotch, maple board, dot inlays (Chase's friend, phone)) | 24 | 1.59 / 1.03 | 4.5 | 3.01 | 1.000 | 0.951 | 1.000 |
