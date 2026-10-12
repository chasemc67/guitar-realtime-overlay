# CoreML export + Apple-silicon real-time tracker (model v1), 2026-10-11

Setup:
- Hardware: Mac Mini M4 (24 GB, macOS 27.0.1).
- Environment: `~/gro-coreml` venv (python 3.12, coremltools 9.0, torch 2.7.0, ultralytics 8.4.175, numpy 2.2).
- Model: v1 yolo11s-pose `best.pt`.
- Module: `src/gro/infer/` (see `docs/INFERENCE.md`).
- Scripts: `scripts/coreml_parity.py`, `scripts/bench_coreml.py`, `scripts/live_cam.py`.
- Raw results: `eval/coreml_v1/*.json`.

## 1. Export

`YOLO(best.pt).export(format="coreml", imgsz=640, nms=False)` produces an mlprogram with a 640×640 RGB image input
(1/255 baked in) and a single `(1, 155, 8400)` float32 output. The tensor layout is in INFERENCE.md.

| Variant | Size | Notes |
|---|---:|---|
| `v1_fp16.mlpackage` (`half=True`) | 22 MB | **Recommended.** Runs on the ANE. |
| `v1_int8.mlpackage` (`int8=True`, 8-bit k-means palettized weights) | 11 MB | Same speed as fp16 on the ANE. Slightly larger keypoint error. |
| `v1_fp32.mlpackage` | 43 MB | Runs on the GPU only (the ANE needs fp16). Matches torch exactly. |

Export pitfalls:
- coremltools 9.0 fails with torch 2.14 / numpy 2.5 (`TypeError: only 0-dimensional arrays can be converted`).
  Pin `torch==2.7.0` and `numpy<2.3`.
- int8 needs scikit-learn.

## 2. Parity (210 frames: 70 spread over each of 09-06 [mirrored, 720p], Stan [mirrored, 1080p], nBpe2 [1080p])

The reference `torch` is the `.pt` run as an nn.Module on the *same* 640² letterbox. `ultralytics` is
`YOLO.predict` with a rect letterbox (640×384 for 16:9), the path the demo v2 caches used. Keypoint px are
source-frame px over slots with conf > 0.5 in both. The wire columns compare the per-frame fit (no smoothing, frets
24/22/22): the distance between wire midpoints, over all wires.

| Candidate vs ref | kpt px mean / p95 / max | \|Δconf\| mean / max | anchor flips / frame | wire px mean / p95 / max | fit lost |
|---|---|---|---:|---|---:|
| **CoreML fp16 vs torch** | 0.98 / 2.09 / 14.1 | 0.004 / 0.84 | 0.09 (of 43.5) | **0.93** / 2.07 / 35.4 | 0 |
| CoreML int8 vs torch | 1.57 / 3.31 / 11.1 | 0.004 / 0.24 | 0.04 | 0.91 / 2.21 / 28.9 | 0 |
| CoreML fp32 vs torch | 0.00 / 0.00 / 0.00 | 0 / 0 | 0 | 0.00 / 0.00 / 0.001 | 0 |
| torch (640²) vs ultralytics (rect) | 0.38 / 1.59 / 18.0 | 0.004 / 0.82 | 0.12 | 0.45 / 1.68 / 69.8 | 0 |
| CoreML fp16 vs ultralytics | 1.16 / 2.70 / 17.4 | 0.006 / 0.84 | 0.19 | 1.19 / 2.92 / 71.1 | 0 |

Per clip, fp16 vs torch:

| Clip | kpt px mean / p95 / max | wire px mean / p95 / max |
|---|---|---|
| 09-06 (720p) | 0.77 / 1.44 / 3.7 | 0.70 / 1.25 / 5.5 |
| Stan (1080p) | 0.98 / 2.08 / 14.1 | 0.94 / 2.15 / 35.4 |
| nBpe2 (1080p) | 1.21 / 2.54 / 10.3 | 1.18 / 2.67 / 17.4 |

Reading:
- fp16 conversion moves keypoints by about 1 source px (≈ 0.3 px at network resolution). That is smaller than the
  per-frame raw-fit jitter in demo v2 (hf 1.16 px Stan, 1.84 px nBpe2; `endclip_jitter_v2.md`), and the One-Euro
  filter absorbs it.
- The tails (max) come from a few frames where one keypoint's confidence crosses the 0.5 anchor threshold (flips) and
  RANSAC settles on a slightly different fit. The rect vs square letterbox change in PyTorch alone produces bigger
  tails (69.8 px), so this is fit fragility on borderline frames, not a conversion problem.
- int8 adds keypoint error (1.6 px mean) but no extra wire error.
- No variant lost a fit relative to its reference.

## 3. Speed on the M4

Model only, batch 1, 640², 200 calls. Times include the PIL wrap + `predict` (Python overhead included):

| Variant | ALL | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|---|---|---|---|---|
| fp16 | **4.25 / 4.94** | 4.25 / 4.57 | 11.7 / 11.8 | 23.0 / 23.2 |
| int8 | 4.10 / 4.37 | 4.10 / 4.36 | 17.7 / 18.2 | 22.9 / 23.1 |
| fp32 | 13.2 / 13.3 | 40.1 / 40.2 | 13.2 / 13.4 | 40.0 / 40.2 |

All cells are median / p95 ms. For reference, PyTorch on MPS takes 23.8 ms and on CPU 106 ms. First load / compile of
fp16 with ALL takes 2.2 s (0.8 s when cached).

End to end, `FretboardTracker.process` with fp16 / ALL, board-end detection on, One-Euro on, 300 frames per clip:

| Frames | decode excluded: fps, median / p95 ms | decode included: fps, median / p95 | per-stage median ms |
|---|---|---|---|
| 720p (09-06, mirrored) | **87.8**, 10.4 / 15.9 | 85.3, 10.7 / 16.4 | model 4.4, post 5.6, mirror 0.15, letterbox 0.16 |
| 1080p (Stan, mirrored) | **86.6**, 9.8 / 14.3 | 81.1, 11.2 / 15.2 | model 4.5, post 4.5, mirror 0.33, letterbox 0.17 |
| 1080p (nBpe2) | **91.0**, 9.1 / 16.3 | 85.1, 9.8 / 17.9 | model 4.4, post 4.4 |

Post-processing is pure numpy. RANSAC + IRLS fit takes ~4.5 ms, and the board-end probe fit + ridge test another
~4.5 ms. With the board-end check on every frame (demo v2 behaviour, `end_every=1`), the pipeline runs at 67–69 fps
(post 9–10 ms). The tracker now checks the board end on every frame only until the fret-count vote has 60 votes, then
on every 10th frame. The count of a guitar is constant, so this saves ~4.5 ms per frame.

The OpenCV reference renderer (`render_state`) takes 12 ms at 720p and 20 ms at 1080p. That is as much as the whole
tracker, so the app should draw from `state.H` on the GPU.

**MacBook expectations** (estimates, not measured):
- The model is ANE-bound. On M1/M2 MacBooks (11–16 TOPS ANE vs 38 on M4), expect ~6–10 ms/frame. M3/M4 MacBooks
  should match the Mini.
- Python post-processing is single-core. Expect ~1.3× slower on M1.
- So expect about 50–70 fps tracker throughput on an M1 Air: comfortable at 30 fps, borderline at 60 fps.
- Fanless Airs may throttle under sustained load. The ANE draws little power, so this matters mostly for the CPU part.
- A Swift port of the post-processing would remove most of the ~5 ms CPU cost.

## 4. Real-time check (`scripts/live_cam.py`, 30 s, video paced at its own frame rate, newest-frame-wins)

| Clip | Source | Tracker only: fps, drops, latency median / p95 | + OpenCV render: fps, drops, latency median / p95 |
|---|---|---|---|
| 09-06 | 720p **60 fps** | **60.0**, 1 / 1800, 14.2 / 22.2 ms | 42.8, 515 / 1800, 19.3 / 28.0 ms |
| Stan | 1080p 30 fps | **30.0**, 0 / 899, 12.5 / 19.1 ms | 29.7, 9 / 899, 9.7 / 32.8 ms |
| nBpe2 | 1080p 30 fps | **30.0**, 0 / 899, 18.1 / 26.9 ms | 29.7, 8 / 899, 10.8 / 31.9 ms |

Latency is measured from frame available to TrackState ready. Notes:
- At 30 fps, per-frame latency is a bit higher than in the batch benchmark (12–18 ms vs ~10). The ANE/CPU idle
  between frames, and the decode thread competes for the GIL.
- The tracker keeps up with 60 fps 720p. The CPU renderer is what drops frames.
- All runs had a valid fit on 100% of processed frames. Fret counts were 24 / 22 / 22, the same as demo v2.
- `live_cam.py --source 0` runs the same loop on a webcam (`--show` for a window, `--hud` for fps/latency). It is not
  run here because the Mini has no camera.

## 5. Demo from CoreML (rendered on the Mini)

- Video: `/workspace/guitar-overlay/demo/coreml/coreml_stan_45s.mp4`. Stan, first 45 s, 1348 frames, H.264 960×540,
  4.4 MB, every frame processed (`--offline`).
- Contact sheet: `demo/coreml/coreml_stan_45s_sheet.jpg` (20 spread frames + 5 consecutive), copy at
  `eval/coreml_v1/coreml_stan_45s_sheet.jpg`.
- Full-size frames: `demo/coreml/frames/`.

What the frames show:
- The grid is on the neck in all 25 sampled frames.
- The nut sits at the headstock and the grid ends before the neck pickup. The fret count is detected as 22.
- No visible jumps across the 5-frame run.

Same open problems as demo v2:
- The overlay is drawn over the fretting hand.
- The blue cells barely show on the maple board.
- The first ~0.5 s draws 24 frets until the vote switches.

## Not done / caveats

- Parity measures agreement with PyTorch, not accuracy. Accuracy is the same as v1 (`model_v1_report.md`,
  `model_v1_0906_honest.md`).
- Speed is measured on the Mini only. The MacBook numbers above are estimates.
- No Swift/Electron code. The port notes are in `docs/INFERENCE.md`.
- The webcam path is untested on real hardware.
