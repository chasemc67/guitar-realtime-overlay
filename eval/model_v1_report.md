# Model evaluation: v1

Errors use original-resolution pixels and absolute slots (vis > 0, prediction conf > 0.5).
D13 headline: perpendicular distance to the GT wire through its bass/treble anchors.
Outer-string error uses a line fitted through each GT side; incomplete wires have no wire error.
Coverage counts GT vis=2 slots predicted with conf > 0.5. Detection uses board confidence >= 0.25.
Numbering: a frame is correct when every prediction lying on a GT wire (within half the local fret spacing)
carries that wire's index; missed detections count as wrong. Index set: predicted fret set == GT labelled (vis>0) set.
Missing metric samples are n/a. Per-frame and per-slot errors are in the JSON.

| Set | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pseudo_gt_test | 185 | 2.35 / 1.28 | 1.79 | 3.34 | 0.976 | 0.989 | 0.951 | 0.254 | 0.892 |
| pseudo_gt_2026-09-06 | 250 | 1.59 / 0.81 | 1.51 | 2.55 | 0.944 | 1.000 | 0.968 | 0.008 | 0.925 |
| hand_gt_heldout | 2 | 1.15 / 0.91 | 2.03 | 2.55 | 0.933 | 1.000 | 1.000 | 0.000 | 0.938 |
| hand_gt_in_train_videos | 6 | 3.25 / 1.68 | 4.31 | 6.23 | 0.915 | 1.000 | 1.000 | 0.000 | 0.868 |

## pseudo_gt_test per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yt_VT4VkOkFIRE | 7 | 3.16 / 2.26 | 1.43 | 3.71 | 0.942 | 1.000 | 1.000 | 0.000 | 0.896 |
| yt_nBpe2YQEzZo | 54 | 4.56 / 1.99 | 3.19 | 6.34 | 0.971 | 1.000 | 0.870 | 0.222 | 0.882 |
| yt_tNAX9maMla8 | 3 | 5.85 / 4.94 | 35.45 | 36.07 | 0.300 | 0.333 | 0.333 | 0.000 | 0.286 |
| yt_vXzydcRC1bA | 121 | 1.32 / 1.07 | 0.95 | 1.78 | 0.998 | 1.000 | 1.000 | 0.289 | 0.911 |

## pseudo_gt_2026-09-06 per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-06_09-00-01 t=000s+ | 30 | 0.94 / 0.73 | 1.20 | 1.67 | 0.941 | 1.000 | 1.000 | 0.000 | 0.939 |
| 2026-09-06_09-00-01 t=030s+ | 31 | 3.55 / 0.78 | 1.48 | 4.49 | 0.941 | 1.000 | 0.968 | 0.032 | 0.934 |
| 2026-09-06_09-00-01 t=060s+ | 29 | 1.32 / 1.02 | 1.83 | 2.46 | 0.941 | 1.000 | 0.966 | 0.000 | 0.914 |
| 2026-09-06_09-00-01 t=090s+ | 30 | 0.95 / 0.70 | 1.47 | 1.91 | 0.950 | 1.000 | 1.000 | 0.000 | 0.937 |
| 2026-09-06_09-00-01 t=120s+ | 30 | 1.10 / 0.77 | 1.35 | 1.93 | 0.940 | 1.000 | 1.000 | 0.000 | 0.926 |
| 2026-09-06_09-00-01 t=150s+ | 29 | 1.94 / 0.75 | 1.50 | 2.87 | 0.935 | 1.000 | 0.862 | 0.000 | 0.922 |
| 2026-09-06_09-00-01 t=180s+ | 30 | 1.01 / 0.81 | 1.59 | 2.06 | 0.959 | 1.000 | 1.000 | 0.000 | 0.946 |
| 2026-09-06_09-00-01 t=210s+ | 30 | 2.04 / 0.96 | 1.74 | 3.14 | 0.947 | 1.000 | 0.967 | 0.033 | 0.908 |
| 2026-09-06_09-00-01 t=240s+ | 11 | 1.23 / 0.91 | 1.43 | 2.07 | 0.930 | 1.000 | 0.909 | 0.000 | 0.859 |

## hand_gt_heldout per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-06_09-00-01_10s | 1 | 1.01 / 0.84 | 0.85 | 1.44 | 0.958 | 1.000 | 1.000 | 0.000 | 0.958 |
| 2026-09-06_09-00-01_60s | 1 | 1.29 / 1.28 | 3.27 | 3.72 | 0.905 | 1.000 | 1.000 | 0.000 | 0.917 |

## hand_gt_in_train_videos per clip

| Clip | Frames | D13 mean / median px | Outer mean px | Kpt mean px | Coverage | Detection | Numbering | Index set exact | Index Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-30_15-06-39_10s | 1 | 0.99 / 0.66 | 1.78 | 2.07 | 1.000 | 1.000 | 1.000 | 0.000 | 0.609 |
| 2026-09-03_20-38-02_10s | 1 | 3.31 / 1.12 | 1.87 | 4.43 | 0.920 | 1.000 | 1.000 | 0.000 | 0.920 |
| 2026-09-03_20-38-02_60s | 1 | 6.80 / 7.17 | 3.20 | 8.02 | 0.864 | 1.000 | 1.000 | 0.000 | 0.880 |
| I4Vt5Q4aUrw_08 | 1 | 1.42 / 0.85 | 4.48 | 5.09 | 0.955 | 1.000 | 1.000 | 0.000 | 0.955 |
| IZJdHyfkrsg_08 | 1 | 4.67 / 3.67 | 10.15 | 11.46 | 0.857 | 1.000 | 1.000 | 0.000 | 0.929 |
| mP2wtBlcCSc_08 | 1 | 2.07 / 1.44 | 6.23 | 7.22 | 0.909 | 1.000 | 1.000 | 0.000 | 0.917 |

Batch-1 timing: square imgsz, 20 warmups, 200 iterations; forward-only unless *_end_to_end.
```json
{
  "pytorch_fp16_cuda": {
    "fps": 130.19867616857118,
    "ms": 7.680569645003743,
    "iterations": 200,
    "warmup": 20
  },
  "onnx_cuda": {
    "unavailable": "CUDA provider failed to initialize"
  },
  "tensorrt_fp16": {
    "unavailable": "requires sibling/--engine TensorRT engine"
  },
  "pytorch_fp16_end_to_end": {
    "fps": 93.97268744451578,
    "ms": 10.64138982500026,
    "iterations": 200,
    "warmup": 20,
    "input": [
      720,
      1280,
      3
    ]
  }
}
```

Skipped hand frames: 0.
