# Real-time inference (`gro.infer`)

`gro.infer` turns camera frames into a smoothed fretboard homography. It is the demo v2 pipeline (model → keypoints →
board-end vote → RANSAC fit → One-Euro) packaged as one stateful class with swappable model backends. It is written
for the Jam Dashboard desktop app. Dependencies: numpy and opencv-python, plus one runtime: `coremltools` (macOS,
`.mlpackage`), `onnxruntime` (`.onnx`), or `ultralytics` + `torch` (`.pt`). The package does not import anything from
`scripts/`.

## Quick start

```python
from gro.infer import FretboardTracker, render_state
from gro.theory import scale_positions

tracker = FretboardTracker("v1_fp16.mlpackage", mirrored=False, compute_units="ALL")
notes = scale_positions("A", "minor pentatonic")
while True:
    ok, frame = cap.read()                      # BGR uint8, any size
    state = tracker.process(frame, t=capture_time_s)
    out = render_state(frame, state, notes=notes)  # or draw it yourself from state.model / state.wires()
```

Use one tracker per camera or clip. Call `tracker.reset()` on a scene cut or a guitar change, because the fret count
vote and the smoother are per clip.

Runnable examples:
- `scripts/live_cam.py --model v1_fp16.mlpackage --source 0 --show [--mirrored]` runs on a webcam.
- `--source clip.mp4` plays a file at real-time pace and drops frames when the tracker falls behind.
- `--offline --out x.mp4` processes every frame and renders an H.264 file.

## Constructor

| arg | default | meaning |
|---|---|---|
| `model` | — | path (`.mlpackage` / `.mlmodel` / `.onnx` / `.pt`) or a `Backend` instance |
| `mirrored` | False | D1. The frame shows a lefty unmirrored, or a righty through a mirrored camera. The tracker flips internally; all outputs are in original-frame pixels. |
| `frets` | None | Known fret count, which skips board-end detection. None votes it per clip (starts at 24). |
| `fps` | 30 | Nominal rate, used only when `process` gets no `t` |
| `smooth` | True | One-Euro on the homography (`min_cutoff=0.4`, `beta=3`, `speed_floor=2` fret widths/s) |
| `kpt_thr` / `box_thr` | 0.5 / 0.25 | Keypoint confidence for an anchor; board score for a detection |
| `hold` | 10 | Frames to keep the last fit when the detection is unusable |
| `end_every`, `settled_votes` | 10, 60 | After 60 board-end votes, check the board end only on every 10th frame (saves ~4.5 ms/frame). `end_every=1` gives the exact demo v2 behaviour. |
| `compute_units` (CoreML) | `"ALL"` | `ALL` / `CPU_AND_NE` / `CPU_AND_GPU` / `CPU_ONLY` |
| `device` (torch) | auto | `cuda` / `mps` / `cpu` |

## `TrackState` (all coordinates are original-frame pixels)

| field | |
|---|---|
| `valid` | A fit is available (it may be held from a previous frame; see `held`) |
| `model` | `gro.geometry.FretboardModel`. `model.H` (3×3) maps canonical plane points `P(u, y) = (u, (y − 0.5)(1 + τu))` to image px, with τ = `model.tau` = 0.30, u = 1 − 2^(−fret/12), and y = 0 on low E, 1 on high E. Helpers: `project_fy(fret, y)`, `fret_line(n)`, `string_line(s)`, `cell_quad(n)`. |
| `H` | Shortcut for `model.H` (None if not valid) |
| `wires()` | `(n_frets+1, 2, 2)` fitted wire endpoints `[low-E end, high-E end]`. Wire 0 is the nut. |
| `n_frets` | Voted (or overridden) last fret |
| `kpts` | `(50, 3)` raw model keypoints x, y, conf. Row `2·fret + side` (side 0 = low-E edge, 1 = high-E edge). NaN when no board was detected. |
| `anchors` | Keypoints with conf > `kpt_thr`, as `gro.anchors.Anchor` (canonical, i.e. un-mirrored, coordinates) |
| `board_score` | Detection score of the board box |
| `end_estimate` | This frame's board-end evidence (None = no evidence, or not checked this frame) |
| `held`, `rmse_px` | Fit was held; RMSE of this frame's raw fit |
| `timings_ms` | `mirror`, `letterbox`, `model`, `post`, `total` |

The stages are also public: `canonical(frame)` → `infer(canon)` → `update(canon, kpts, t)`. An app can run model
inference on one thread and the post-processing on another.

## Model files

Export from the training checkpoint (macOS, coremltools 9 needs torch ≤ 2.7 and numpy < 2.3):

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="coreml", imgsz=640, half=True, nms=False)   # fp16, 22 MB, recommended
YOLO("best.pt").export(format="coreml", imgsz=640, int8=True, nms=False)   # 8-bit palettized, 11 MB
```

Model v3 (FretNet, permissive, no Ultralytics; `eval/model_permissive.md`) has the same interface with one anchor:
`python scripts/export_fretnet.py runs/fretnet/v3b_r50/best.pt --coreml v3_fp16.mlpackage` writes an image input
`image` and one output `(1, 155, 1)`, and `FretboardTracker("best.pt")` loads a FretNet checkpoint directly.

Weights are not in git. v1 = `runs/pose/v1/weights/best.pt` on the PC; the `.mlpackage` files live in
`~/gro-coreml/weights/` on the Mac Mini.

### Tensor layout (for a Swift / Electron port)

- **Input** `image`: CoreML image type, 640×640 RGB. The 1/255 scaling is baked into the model, so pass a
  `CVPixelBuffer` (BGRA/32ARGB) directly. For ONNX: `images` float32 `(1, 3, 640, 640)`, RGB in [0, 1].
- **Letterbox** (must match training/ultralytics exactly):
  - `r = min(640/h, 640/w)`, `nw = round(w·r)`, `nh = round(h·r)`, bilinear resize
  - `left = round((640 − nw)/2 − 0.1)`, `top = round((640 − nh)/2 − 0.1)`, pad value 114 (gray)
  - Example: 1920×1080 → 640×360 at top = 140.
  - Do the D1 mirror flip *before* the letterbox.
- **Output** (one float32 multiarray, name auto-generated, e.g. `var_1348`): shape `(1, 155, 8400)`, channel-major.
  - 8400 anchors = 80² + 40² + 20² (strides 8/16/32).
  - Per anchor: `[cx, cy, w, h]` box in letterboxed px, `score` (sigmoid, single class "fretboard"), then 50 × `(x, y, conf)`
    keypoints in letterboxed px. Keypoint conf is already sigmoid-ed.
  - Keypoint k = `2·fret + side`, frets 0..24, side 0 = low-E (bass) edge, 1 = high-E (treble) edge.
- **Decode**: take `argmax(score)` over the 8400 anchors. With one board per frame this is the box NMS would keep, so no
  NMS is needed. Reject if score < 0.25. Unletterbox: `x = (x − left)·w/nw`, `y = (y − top)·h/nh`, then clamp to
  the frame (`clip_kpts`, which matches ultralytics). If mirrored, compute `x' = W − 1 − x` on output.

### Post-processing to port

All of it is in numpy, ~5 ms/frame in CPython on an M4. A Swift port would be well under 1 ms.

1. Anchors = keypoints with conf > 0.5.
2. `gro.geometry.fit_fretboard`: RANSAC on 4-point samples (300 iterations, inlier threshold 0.15 × local fret width) +
   Huber IRLS refinement of an 8-DOF homography, with fixed taper τ = 0.3.
3. `gro.board_end` (only when the fret count is unknown):
   - Probe fit on slots ≤ 21.
   - Rectify the high-fret strip and test wires 19..24 for two-sided ridges.
   - Feed the result into `FretCountVote` (mode with hysteresis).
4. `gro.smooth.OneEuroModel`:
   - One-Euro filter on 10 control points (wires 0/6/12/18/last × outer strings) with one shared cutoff.
   - DLT refit of H.
5. Render from H. Everything is drawn in canonical (fret, string) space and projected, so a GPU renderer only needs
   H, τ and `n_frets`.

## Speed (Mac Mini M4, v1 yolo11s-pose, 640²; details in `eval/coreml_v1.md`)

- Model: fp16 on the Neural Engine takes **4.3 ms** median. CPU only takes 23 ms. PyTorch on MPS takes 24 ms.
- End to end (`process`, board-end vote on): **~10 ms → ~87 fps** at 720p and at 1080p.
- The OpenCV reference renderer adds 12 ms at 720p and 20 ms at 1080p. The app should render on the GPU from `state.H`.
