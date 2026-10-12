# Guitar Real-Time Overlay — Technical Plan

Status: draft v4 (2026-10-11), which adds the owner decisions below (D1–D12; D13 provisional). Owner: Chase. This is an opinionated plan; open
questions are collected at the end (§10).

## Decisions (2026-10-11)

These decisions come from Chase (owner) and override anything below that conflicts with them.

**D1. Left-handed support is required in v1, as a UI toggle. It is never inferred from video.**
- The model assumes a fixed **right-handed (canonical) orientation**. There is no handedness classifier, head, or label,
  and nothing in the pipeline tries to detect handedness.
- When the toggle is on, the app horizontally mirrors the input frame before inference. Mirroring is folded into the
  Stage 1 resize and the Stage 2 crop affine, so it costs nothing. Results map back by composing the mirror into the
  fit: `H_img = M · H_canon`, where `M = [[−1,0,W−1],[0,1,0],[0,0,1]]`. Equivalently, the anchors are mirrored for
  rendering. The fit, the filter, and KLT all run in canonical space; only the renderer sees `M`.
- The toggle is really "mirror input", so it also fixes a right-handed player seen through a mirrored (selfie) camera
  feed. The UI labels it "Left-handed / mirrored camera".
- **Data.** Left-handed footage is not annotated separately. Left-handed source clips, and clips whose creator mirrored
  the video, get one clip-level `mirrored: true` flag during the ingest yes/no pass (§4). They are mirrored before frame
  extraction and labeled like any other right-handed clip.
- **Where the flag lives (2026-10-11).** It's the `mirrored` column (yes/no) in `data_manifests/*.tsv`:
  `yt_batch0*.tsv` for YouTube clips (all `no` so far) and `owner_videos.tsv` for Chase's videos and `data/test_ext`
  clips (Chase's 08-30/09-03/09-06 and Stan's Telecaster are `yes`). An optional `frets` column holds a known fret count
  (blank = detect the board end). `gro.auto.manifest.clip_info(video)` reads both by file stem (with or without the
  `yt_` prefix). `demo_model.py` and `cache_preds.py` use the flag automatically, and `eval_model.py` uses it for owner
  and hand-GT frames, re-mirroring stored GT if its label file disagrees. In all three, `--mirrored` /
  `--no-mirrored` overrides it, and in the demo `--frets N` overrides the fret count. The flag in label JSON
  (`video.mirrored`) records the frame its keypoints were stored in.
- **Flip augmentation must keep the canonical orientation.** A plain horizontal flip of a right-handed frame produces a
  left-handed-looking frame. Physically correct labels on that frame would teach the model the orientation we just
  removed. So h-flip = mirror the x coordinates **and swap bass↔treble for every slot** (YOLO `flip_idx` =
  `[1,0,3,2,…]`). This keeps the geometry consistent with the canonical orientation. String-gauge order and hand pose
  become physically wrong, so keep h-flip probability low (≤ 0.2) and ablate it in M4. A unit test checks that the
  flipped labels fit an `H` with the same orientation (sign of the determinant) as unflipped canonical labels.

**D2. First target platform: a desktop webcam on Chase's Apple Silicon MacBook.**
- **Inference.** CoreML (`.mlpackage`, ML Program, fp16) converted with `coremltools`. It runs on the ANE or the GPU,
  chosen by benchmarking `compute_units` = `CPU_AND_NE` / `CPU_AND_GPU` / `ALL`; on a Max-class GPU, the GPU may beat
  the ANE for this model size. PyTorch MPS is for quick in-the-loop checks on the Mac, not the shipping path.
- **Latency target on M-series:** ≤ 10 ms p95 compute per frame, at 60 fps if the camera supports it (§2.5).
- **Early prototypes** can run on the Windows RTX 4090 PC with a webcam: PyTorch CUDA → ONNX → TensorRT / ONNX Runtime.
  The same app code is used, with a pluggable inference backend (`coreml | tensorrt | onnxruntime | torch`).
- **Web (WebGPU) and iOS come later** and are out of v1 scope.
- **Priority: get it working well first.** No porting work starts until the §6 targets are met on the Mac.
- **Consequences.**
  - CoreML conversion runs anywhere, but CoreML *prediction* (parity tests, `bench.py`) needs macOS. That puts the Mac
    in the eval loop.
  - Keep the rotated-crop sampling (`grid_sample`/affine warp) out of the model graph, and do it in vImage, Core Image,
    or Metal. Use fixed input shapes (1024×256), and check with the Xcode Core ML performance report (or
    `MLComputePlan`) that no layer falls back to the CPU.

**D3. Guitars: electric only for now; all guitar types eventually.**
- v1 data, training, and eval target 6-string electrics (Strat/Ibanez/LP/Tele-style, 21–24 frets, any inlay style
  including ornate vines). Acoustic and classical clips may still be collected opportunistically, but they are tagged and
  held out of the v1 training mix and acceptance metrics.
- The diversity quota in §4 is overridden for v1: ~100% electric. Keep the inlay/wood/lighting/camera quotas.
- Nothing in the label schema or geometry is electric-specific (`guitar.type` is already a field), so widening to
  acoustic/classical later is a data job, not a redesign.

**D4. Fret coverage: every fret visible in the camera, resilient to partial necks, including the nut NOT visible.**
- The overlay must cover every fret wire in view (up to 22/24), not a fixed 0–15 window. Drawing range becomes
  `[min_visible_fret − 1, max_visible_fret + 1]`, clipped to the frame and to `guitar.frets`.
- **The anchor representation must resolve the fret index without the nut.** Fret slots are still named by absolute
  index (`f1..f24`), but the fit must not assume the nut is present. Index is resolved from:
  1. ~~Fret-spacing ratios~~ — **do not work on their own (found in M1).** Fret positions are a geometric
     progression from the bridge: `u_{n+s} = 1 − 2^(−s/12)·(1 − u_n)`, so an index shift is an *affine* map of `u`
     that any homography absorbs. Cross-ratios of fret positions are identical for every shift. Only two weak
     geometric signals remain: the neck taper, and the *metric* shape of the board (fret gap ÷ board width changes
     ~6% per fret, observable through `H` with roughly known camera intrinsics). On synthetic pinhole data the metric
     cue is right ~90% of the time (`resolve_index_shift(image_size=…)`), but on the real M1 frames it was biased by
     2–3 frets (nut width/scale uncertainty, lens distortion, binding vs. edge). Treat it as a tie-breaker at most.
  2. **Inlay pattern (primary cue).** The single/double-dot pattern (3,5,7,9,12,15,…) or, for ornate inlays, the 12th-fret
     "double"/larger motif and the inlay-to-inlay spacing. The model gets an inlay head (§2.2) whose detections vote on
     `s`.
  3. **Landmarks:** body joint/neck heel (≈ fret 16–17 on most electrics), last fret (`guitar.frets`), headstock.
  4. **Per-session lock + hysteresis** (§2.4): once confident, the index is locked and only re-opened after track loss.
- Labels for nut-less frames are fine: the annotator types the index of one wire (counted from the 12th-fret marker or
  the last fret) and the tool fills the rest.
- **First PoC exception:** the M1/M2 proof-of-concept may require the nut to be visible (index = count from the nut).
  The fit library still implements the shift search from day one so the PoC isn't a dead end.

**D5. No hand-occlusion handling needed. Drawing the overlay over the fingers is fine.**
- The M6 "hand-occlusion mask" item is dropped (optional polish at most). The renderer draws on top of everything.
- Occlusion still matters for *detection* (vis = 1 labels, aux line head, hand-paste augmentation); only the
  *rendering* requirement is removed.

**D6. Data policy: downloading YouTube/TikTok clips for internal training is OK. No commercial release is planned.**
- `tools/collect.py` (yt-dlp) is approved for YouTube, TikTok, and Instagram reels. Raw media still stays out of git and
  out of public demos.
- Keep `source_url`, `uploader`, and `license` per clip anyway (cheap, and keeps takedowns and a future retrain-on-owned
  path possible), but CC filtering is no longer a priority.

**D7. The RTX 4090 PC can be used any time, including unattended overnight runs.**
- Long queued runs and overnight sweeps (`queue.sh`) are fine. Auto-resume stays, for crashes and reboots.
- Wake-on-LAN / Tailscale still needs a one-time setup; it's no longer a policy question.

**D8. Annotation quality is critical and needs Chase's review. Every annotation batch produces review images.**
- Labels are the ceiling on everything downstream (index errors especially, D4), so no batch counts as done until
  Chase has reviewed it. Chase reviews from screenshots, so the review artifact is a set of images, not a web app.
- `scripts/review_sheet.py <labels_dir> --batch <name>` (`gro.review`) writes `review/<batch>/`:
  - `contact_sheet_NN.jpg`: 3×3 tiles (configurable, ~2000 px wide), one per annotated frame, cropped to the board.
  - `frames/<clip>_fNNN.jpg`: full-res frame with the clicked anchors (labelled `<fret><b|t>`, green = visible,
    orange = occluded, red X = rejected by the fit) and the refitted grid (every fret wire and string; solid inside
    the annotated range, dashed where extrapolated).
  - `strips/<clip>_fNNN.jpg`: neck-aligned 3× zoom crops, raw above fit, to judge wire-by-wire alignment.
  - `summary.csv`, plus a caption bar on every image: file, frame, inlier mean/max px error (and which anchor),
    rejected points, nut visible, mirrored, τ. Captions turn orange when max error > 3 px or a point was rejected.
- Commit `review/<batch>/` when the batch's images are < 5 MB in total. Otherwise gitignore the batch and commit only
  the contact sheets.
- A web viewer comes later, and only if the contact sheets stop scaling.
- First batch: `batch-000-hand` (the 3 M1 frames). Inlier mean error is 0.55–0.64 px. The worst point is 4.65 px, at
  the nut's treble anchor (`0t`) on 2026-09-03; it is flagged and waiting on Chase's review.

**D9. Storage: keep the box light.**
- Raw and scraped clips live on the Windows PC (lots of disk) and/or in the `Training data` subfolder of Chase's
  Google Drive folder. This overrides "Google Drive is the archive" in §5: either location
  is fine as the archive.
- The box (this dev machine) keeps only working sets: the clips and frames of the batch being annotated or reviewed.
  Delete them once they're synced back.
- Git holds code, labels, manifests, and small review images (D8), never raw media.

**D10. Data collection blocker: the box can't download from YouTube or TikTok.**
- From the box, YouTube rejects every yt-dlp player client: HTTP 429 and "Sign in to confirm you're not a bot". The
  box has a datacenter IP. yt-dlp's TikTok extractor is currently marked broken.
- Plan: run `tools/collect.py` (yt-dlp) on the Windows PC or another home machine (residential IP), writing to the
  PC's disk / Drive `Training data` (D9). Only the working set is synced to the box.
- M1 learnings that shape annotation are in D4 and §1.2: fret spacing can't pin down the fret index. When the nut is
  hidden (fret wrap on Chase's guitars), the index has to come from inlays, the body joint, or the last fret.

**D11. Absolute fret numbering without the nut: use the inlay pattern (Chase, 2026-10-11).**
When the nut/headstock isn't visible, anchor absolute fret numbering on the inlay sequence 3-5-7-9-12: the double-width gap between the 9 and 12 markers stands out, and the 12th is often a double dot. The model should predict per-fret inlay presence (single/double/none) and the fitter matches that sequence against the fitted fret grid. Fret counts vary by guitar (21/22/24), so neck-body-joint and last-fret cues are secondary and less reliable. Accept that numbering may be impossible without inlays or a visible nut (e.g. blank boards, heavy occlusion); degrade gracefully by showing relative positions (pattern shape without absolute notes) or asking the user which fret a marker is on, and keep that answer per session.
Annotation review of batch-000-hand: Chase says it looks great (2026-10-11). Collection runs on the Mac Mini (home IP) until the PC is back.

**D12. Anchorability rule (refines D11, Chase 2026-10-11).** Absolute fret numbering requires at least one visible cue: (a) the headstock/nut, (b) the 9-12-15 inlay frets, or (c) all of the 3-5-7 inlay frets. Otherwise fall back to relative positions. Every annotated frame carries `anchorable: bool` and `anchor_cue: "nut" | "inlay_9_12_15" | "inlay_3_5_7" | null` (first matching cue in that order); review sheets show the flag in each caption, and eval reports absolute-numbering accuracy only on anchorable frames.

**D13. Anchors sit on the outer strings, not the board edges (provisional, 2026-10-11; pending Chase on Q16).** Each
anchor is where a fret wire crosses an outermost *string*: the low-E string on the bass side and the high-E string
on the treble side. Canonical y = 0 / 1 are the low-E / high-E strings (`string_y` margin 0), and the drawn board
extends `BOARD_PAD` = 0.1 (half a string spacing) beyond them. Why: the overlay needs the strings, strings are visible
against dark shirts where the bass board edge isn't, and it no longer matters whether the side of the neck shows.
Board edges become optional derived fields (`board_edge_kpts` on the re-derived GT frames). All GT in
`samples/anchors` and `samples/anchors_extra` was re-derived this way (`scripts/rederive_gt_strings.py`; check crops
in `review/d13-gt-strings/`). Before that, *neither* GT side followed one convention. The treble side sat at the wire
end (board edge), 1–6 px outside the high-E. The bass side sat at a wire-ridge dip that was usually one string in (on
the A string, 4–19 px from the low-E), not on the low-E. The old note that "GT bass = low-E" was wrong. On the
09-06 guitar, a bright line along the bass edge continues past the board end onto the body, which shows it's the
low-E and not binding (`review/d13-gt-strings/`). Caveat: the re-derived GT uses the same string detector as the
auto pipeline, applied to the GT wire lines. Along-wire GT/auto agreement is therefore partly self-consistency, and
only Chase's review of the crops makes it independent. The perpendicular-to-wire error stays independent.

**D13 confirmed and narrowed (Chase, 2026-10-11).** The outer strings (low E, high E) define the fretboard's height extent instead of the wood edges. The PRIMARY goal is accurate fret wire/marker anchoring and numbering. No individual string detection is required: assume 6 evenly spaced strings between the outer extents (the current joint six-string fit is only used to locate the two outer extents). Outer-extent precision matters less than fret-wire precision, so the eval headline metric is the along-neck / fret-wire error (anchor distance from the GT wire line, "across-wire" in eval_auto output), reported separately from the along-wire (outer-extent) error. Q16 is answered.

## 0. Goal & constraints

Draw a scale overlay onto a guitar fretboard from a live close-up camera, so that it matches `docs/reference-overlay.jpg`:
semi-transparent cells between adjacent fret wires across all 6 strings, thin string lines, white inlay dots,
highlighted target-note cells, correct perspective (trapezoids shrinking toward the body), stable across frames,
every fret that's visible in the camera (D4), including partial necks with the nut out of frame.

Hard constraints observed so far:

- **Scene**: home-office practice, guitar fills the frame, webcam-ish angle, neck diagonal or horizontal, fretting hand
  heavily occludes the frets it plays over (exactly where the overlay matters most).
- **Off-the-shelf models failed** on accuracy and speed, so we train a small custom model.
- **Owner's guitars**: Ibanez-style electrics with ornate vine inlays. A naive "find the dots" heuristic will fail, and
  inlays must not be treated as fret wires.
- **Video**: 60 fps, 720p to ~950p. Budget latency for 60 fps and accept 30 fps model inference with propagation in
  between (§2.5).
- **Training compute**: an RTX 4090 Windows box over ssh, usable any time including overnight (D7). Modal is a paid
  fallback to avoid.
- **Guitars**: electric only for v1 (D3).
- **Runtime target**: Chase's Apple Silicon MacBook with a webcam, using CoreML on the ANE or GPU. The 4090 PC with a
  webcam is used for early prototypes (D2).
- **Handedness**: the model is right-handed only. Left-handed players use a mirror toggle (D1).

## 1. Anchor representation

### 1.1 Options

**A. Raw keypoints.** Fixed slots: the nut and each fret wire `n = 1..24`, with one point where the wire meets the
bass-side fretboard edge (low E) and one where it meets the treble-side edge (high E). That is 50 slots, each with a
visibility flag. (D13: in the labels these points are where the wire crosses the low-E / high-E *strings*.)

**B. Parametric model.** A homography `H` maps fretboard-plane coordinates to the image, plus a fret-position model.
Plane coordinates are:

- `u` = distance from the nut ÷ scale length.
- `v` = across-neck position, from −1 (bass edge) to +1 (treble edge).

Fret `n` sits at `u_n = 1 − 2^(−n/12)`, so `u_12 = 0.5`. The edges taper linearly: half-width
`w(u) = w0 · (1 + τ·u)`, where `τ` is a per-guitar taper constant, roughly 0.25 to 0.35 for 43→56 mm. The free
parameters are `H` (8 DOF) plus `τ`. The scale length is absorbed into `H`, because `u` is normalized.

| | A: raw keypoints | B: homography + fret model |
|---|---|---|
| Learnability | Direct heatmap target. Easy to supervise. | Regressing `H` directly is brittle and generalizes poorly. |
| Occlusion / out of frame | Visibility flags. The rest is missing. | Can draw frets that aren't visible (behind the hand, off-frame). |
| Physics | None. Fret 7 can land between 5 and 6. | Spacing is exact by construction. |
| Annotation cost | 50 clicks per frame. | About 8 clicks per frame (4+ fret lines), then a fit. |
| Rendering | Needs interpolation. | Every cell and string comes out in closed form. |

### 1.2 Recommendation: predict A, fit B

- The **network predicts raw keypoints** (A), as heatmaps with visibility. This is the label format and the training
  target.
- A **robust geometric fit** turns those keypoints into the parametric model (B) every frame. This is the runtime state,
  and it is what the renderer and the tracker consume.
- The **annotation tool uses the same fit** to auto-complete partial clicks (§3).

The fit, per frame:

1. **Collect candidates.** Take keypoint peaks with confidence above a threshold. Each slot `k` gives a known plane
   point `(u_n, ±w(u_n))`.
2. **Fit with RANSAC.** Fit a DLT homography with RANSAC: minimal sample of 4 points, at least 2 fret lines; threshold
   ≈ 0.15 × local fret width. Refine with Levenberg–Marquardt on inlier reprojection error, with a Huber loss. Weight
   each point by its confidence.
3. **Check the fret index (off-by-one guard).** Index errors are the classic failure: the nut is off-frame and the
   network shifts every slot by one.
   - Re-fit with every slot shifted by `s ∈ {−2…+2}` (wider, e.g. `−6…+6`, when the nut is not visible; D4) and keep
     the lowest-cost shift.
   - **Correction (M1):** fret spacing does *not* disambiguate the index. An index shift is an affine map of `u`
     (frets form a geometric progression from the bridge), so every shift fits equally well up to the weak taper and
     metric-shape cues (D4). The absolute index has to come from the nut, inlays, or landmarks (body joint, last fret).
   - Tie-breaker: inlay positions (3, 5, 7, 9, 12, 15, 17…) from the auxiliary inlay head (§2.2).
4. **Fit `τ` per session**, not per frame. Use a running median over high-confidence frames, initialized to 0.3.
5. **Ignore radius and fret height** for v1. The fretboard radius (9.5–17″) moves points by ≲2 px at our scale. A
   per-string `v` offset can be added later if the cell IoU error budget says so.

Visibility and occlusion:

- Each label slot has a state: `2` = visible, `1` = occluded but inferable (behind fingers; the annotator places it
  using the wire on either side of the hand), `0` = out of frame / not present (e.g. fret 24 on a 22-fret neck).
- Training: loss on states 2 and 1, with a lower weight on 1. A separate visibility logit head is trained on all three
  states.
- Runtime: frets beyond what's visible are extrapolated by `H`, but drawn only within `[min_visible_fret − 1,
  max_visible_fret + 3]` and inside the frame. Cells far outside the observed range get faded alpha.

**Name the keypoints by physical identity (bass/treble side), never by image top/bottom.** Labels always live in the
canonical right-handed orientation (D1), so rotation never changes their semantics. In the nut-left crop, the bass edge
is always on the same side, so the model never has to resolve handedness. The one place semantics change is the h-flip
augmentation, which swaps bass↔treble (D1, §3.4).

## 2. Model approach

### 2.1 Pipeline

```
frame ─► [mirror if left-handed toggle on — folded into resize/crop affine, D1]
      ─► [Stage 1] neck OBB detector (full frame, 640px, every ~10th frame or on track loss)
      ─► rotated crop aligned to neck axis (1024×256, nut-side left)
      ─► [Stage 2] keypoint heatmap net (50 kpts + aux heads)
      ─► robust fit (H, τ, index shift) ─► temporal filter ─► renderer (draws with M·H if toggle on)
         ▲                                       │
         └── between model frames: KLT optical-flow homography propagation
```

In steady state, Stage 1 is skipped: the crop for frame t+1 comes from the filtered `H` at frame t, padded by 15%.
Stage 1 runs only to (re)acquire the neck. Everything between the mirror and the renderer works in canonical
(right-handed) image space.

### 2.2 Models

**Baseline (M2): YOLO11s-pose.**
- One class (neck), 50 keypoints, 640px full frame.
- Gives a working model in days, plus ONNX/TensorRT/CoreML export for free.
- Expected weaknesses: regressed keypoints are coarse on a thin diagonal object, and fret cells near the body are only
  ~12–20 px wide at 720p. Treat it as the benchmark to beat, not the destination.

**Target (M3+): custom heatmap + offset network on the rotated neck crop.**

The rotated crop roughly quadruples effective resolution versus letterboxing a diagonal neck.

- Backbone: RepViT-M0.9 or MobileNetV4-Conv-S (ImageNet pretrained), with a small FPN fused to stride 4. ResNet-18 is
  the fallback for simplicity.
- Input: 1024×256. Output: 256×64.
- Heads:
  - 50 keypoint heatmaps (Gaussian σ ≈ 1.5 output px), plus 2-channel sub-pixel offsets (DARK/UDP-style decoding).
  - 50 visibility logits.
  - **Aux, training only:**
    - 1-channel "fret-wire line" map, which helps the net find wires even under partial occlusion.
    - 2-channel fretboard-edge maps.
    - 1-channel inlay-centre map. Its loss is masked on ornate inlays: label those as a region and don't penalize.
- Size and speed: ~3–6 M params and ~2–4 GFLOPs. Estimated ~1.5 ms on the 4090, ~4–6 ms on an M-series ANE, ~8–12 ms
  on an iPhone 13+ ANE, int8/fp16.

RTMPose (SimCC) is the alternative for Stage 2. It's a strong top-down architecture, but its 1D-classification heads
assume roughly square crops, and mmpose's export story is heavier. Try it only if the custom net plateaus.

**Physics prior in training.**
- Add a differentiable spacing loss: from the soft-argmax keypoints, fit a 1D projective map `u_n → x` along the neck
  axis by weighted least squares (closed form, 3 DOF) and penalize the residual.
- Add a collinearity loss on bass and treble edge points.
- Weight both at ~0.1 of the heatmap loss.
- This mostly fixes "fret 7 between 5 and 6" errors under occlusion.

**Stage 1 neck detector.**
- YOLO11n-obb, trained on boxes derived from the keypoint labels for free: the OBB is the min-area rectangle of the
  visible fret ends, extended to the nut.
- Only needs to be roughly right; the Stage 2 crop is padded.

### 2.3 Input resolution / camera

Capture at native 720p+. Downscale only inside the crop sampler (bilinear, from the full-res frame) so thin wires near
the body survive.

### 2.4 Temporal smoothing & tracking

- **State.** Use the image positions of 4 virtual plane corners: `(u_0,±1)` and `(u_15,±1)`. That's 8 numbers, which
  are well conditioned and intuitive to filter, unlike raw `H` entries. `τ` and the index shift are session-level.
- **Filter.** Apply a One Euro filter per coordinate. Start with `min_cutoff = 1.0 Hz`, `β = 0.02`, `d_cutoff = 1.0`,
  and tune on the jitter metric.
  - Upgrade to a constant-velocity Kalman filter on the 8 params only if One Euro lags visibly on fast neck moves.
  - The Kalman version gives a principled gate via Mahalanobis distance, which is a nice-to-have.
- **Gating.** Reject a measurement if:
  - inlier count < 6, or
  - mean reprojection error > 0.2 × local fret width, or
  - any corner jumps more than 25% of the neck length in one frame.

  On rejection, coast on optical flow. After 15 consecutive rejections, drop the track, fade out the overlay, and
  rerun Stage 1.
- **Index hysteresis.** A change in index shift must win for 5 consecutive frames before it's applied. This prevents
  the whole grid flickering by one fret.
- **Optical-flow propagation**, used when the model runs below the camera rate or a frame is rejected:
  - Track ~100 Shi–Tomasi corners inside the fretboard polygon with pyramidal Lucas–Kanade (KLT), using the
    previous-frame fit as the mask.
  - Exclude a hand region: a cheap skin/hand mask, or simply the region of low keypoint confidence.
  - Estimate an inter-frame homography with RANSAC and compose it onto the state.
  - Cost is ~1–2 ms on CPU at 720p crop resolution.

### 2.5 Latency targets

Primary target (D2): **≤ 10 ms p95 compute per frame on Chase's Apple Silicon MacBook**, which leaves headroom inside
the 16.7 ms budget at 60 fps. The 4090 is the prototype box. Mobile comes later, at ≤ 33 ms with the model at 30 Hz and
flow at 60.

| Stage | MacBook M-series, CoreML ANE/GPU (**v1 target**) | RTX 4090, TensorRT (prototype) | iPhone, ANE (later) |
|---|---|---|---|
| Capture → tensor (mirror + crop + normalize, GPU) | 1 ms | <1 ms | 2 ms |
| Stage 2 model (fp16) | 4–6 ms | 1.5–2 ms | 8–12 ms (every other frame) |
| Stage 1 (amortized) | <1 ms | <1 ms | <1 ms |
| Fit + filter | <1 ms | <1 ms | <1 ms |
| KLT (non-model frames) | 1–2 ms | 1–2 ms | 2–3 ms |
| Render (GPU) | 1 ms | <1 ms | 1–2 ms |
| **Total** | **~7–10 ms** | **~4–5 ms** | **~14–20 ms** |

Two Mac-specific notes:
- Measure the first-inference ANE compile and warm-up separately from steady state, and do it at app start.
- Use zero-copy capture (`CVPixelBuffer`/IOSurface → CoreML) once the app goes native; the Python app (M5) pays one
  copy.

Glass-to-glass latency is dominated by camera and display (~50–80 ms). Measure it once with an LED or a phone-camera
test, but optimize only what we control.

## 3. Annotation format & tooling

### 3.1 Per-frame JSON (one file per clip: `labels/<clip_id>.json`)

```json
{
  "schema": "fretboard-kpts/v1",
  "clip_id": "yt_abc123_0042",
  "video": {"path": "data/clips/yt_abc123_0042.mp4", "w": 1280, "h": 720, "fps": 60, "mirrored": false},
  "guitar": {"type": "electric", "frets": 24, "inlay": "vine", "strings": 6},
  "frames": [
    {
      "t": 1234, "source": "manual|interp|pseudo|synthetic",
      "annotator": "human:chase|agent:cu-01|model:v3",
      "status": "draft|verified|rejected", "qa_reviewer": null,
      "neck_present": true,
      "kpts": {
        "nut":  {"bass": [412.3, 588.1, 2], "treble": [455.0, 540.7, 2]},
        "f1":   {"bass": [480.2, 560.9, 2], "treble": [519.8, 515.0, 1]},
        "f2":   {"bass": [null, null, 0],   "treble": [null, null, 0]}
      },
      "fit": {"H": [[...],[...],[...]], "tau": 0.30, "rmse_px": 1.4, "index_shift": 0},
      "flags": ["motion_blur", "heavy_occlusion"],
      "notes": ""
    }
  ]
}
```

Keypoint triplets are `[x, y, vis]` in full-frame pixels, where vis is 0, 1, or 2 (§1.2). `fit` is recomputed by
tooling and never hand-edited. A converter emits YOLO-pose txt and COCO-keypoints from the same source of truth.

`video.mirrored` is clip-level provenance, set once at ingest (D1). The stored mp4 is the original download. When the
flag is true, the frame extractor and the annotation tool mirror every frame, so all `kpts` are in canonical
(right-handed) coordinates. There is no handedness field: every label is right-handed.

### 3.2 Tool: custom web annotator (not CVAT)

CVAT can do 50-point skeletons, but it can't run our fit live, can't snap to wires, and its UI is hostile to
computer-use agents. Instead, build a small single-page tool: FastAPI backend, canvas frontend, about a week of work.

- **Keyframe workflow.**
  1. Sample keyframes every ~1 s, plus whenever the KLT homography motion exceeds a threshold.
  2. On a keyframe, the annotator clicks the bass and treble ends of **any ≥4 fret wires**, nut included if visible,
     and types or selects the fret index of the first one. Indices of the following ones auto-increment.
  3. The tool runs the §1.2 fit and fills all 50 slots, marked `interp`. It overlays the rendered grid at 50% alpha.
  4. The annotator drags any slot that's off. The fit updates live.
- **Snap.** Each click is refined by searching ±6 px perpendicular to the predicted wire direction for the max of a
  ridge filter. Annotators click near a wire; the tool lands on it.
- **Between keyframes.** Propagate with KLT homography and re-fit, so intermediate frames get `source: interp`. Only
  frames with `rmse_px` below threshold and no flags are exported as training labels. The rest go to a review queue.
- **Navigation.** Keyboard-first. `n`/`p` for frames, `1-9` to set the fret index, `v` to cycle visibility, `f` to
  fit, `a` to accept, `r` to reject.
- **Zoom panels.** Three 4× magnified insets follow the cursor and the last clicked point. This matters for agent
  precision.

### 3.3 Computer-use agent annotation protocol

Agents click poorly at pixel precision but judge overlays well, so the protocol leans on fit, snap, and visual
verification.

1. The tool exposes `/task/<id>`, which shows one frame with a fixed layout and big targets. The agent reads the task
   instructions panel.
2. The agent clicks fret-wire ends in the zoom panel (snap-assisted). It always starts with the most clearly visible
   wire whose index it can determine. It uses the nut, the 12th-fret double inlay, or the body joint, and counts from
   there.
3. The agent presses `f`, screenshots, and checks against a fixed checklist:
   - Every rendered wire lies on a real wire, including behind the fingers (continuity).
   - The grid edges follow the fretboard edges.
   - The 12th-fret line sits on the double-inlay/marker fret.
4. If the check fails, the agent adjusts at most 2 rounds. Otherwise it sets `status: rejected` with a reason.
   **Agents never mark a frame `verified`.** They produce `draft`.
5. **QA loop.**
   - A human (or a second agent with a different prompt) reviews 100% of agent drafts in the first 500 frames, then a
     random 15%, plus every frame with `rmse_px > 2` or an index shift relative to neighbors.
   - Track per-annotator error against a **gold set** of 200 frames labeled by Chase. Re-enable an agent's work only
     while its PCK@0.1 on the gold set stays ≥ 95%.
   - Index errors get zero tolerance and trigger a re-check of the whole clip.
   - Every batch, human or agent, gets `scripts/review_sheet.py` contact sheets, and Chase reviews them before the
     batch is accepted (D8).

### 3.4 Bootstrapping, pseudo-labels, synthetic data

**Pseudo-labels (from M3).**
- Run the current model, the fit, and the tracker over all unlabeled clips.
- Accept frames where:
  - fit `rmse_px < 1.5`,
  - ≥ 8 inliers,
  - index stable over ±30 frames, and
  - forward-backward KLT consistency holds.
- Accepted frames enter training at weight 0.5 and are re-validated each round. Send a 5% sample to the agent QA queue.

**Synthetic data, 2.5D compositing first (cheap, high value).**
- Render a flat fretboard texture procedurally:
  - wood colors (rosewood, maple, ebony, roasted),
  - fret wires with specular highlight,
  - inlays: dots, blocks, sharkfins, trapezoids, **vines/trees of life**, none,
  - fret count 19–24, 6 strings with gauge-appropriate width and sheen.
- Warp it with random `H` sampled from the distribution of fitted real `H`s, onto real background frames with the
  neck removed or from other scenes.
- Paste real fretting-hand cutouts segmented from our clips (SAM-2 masks).
- Labels are exact.
- Target: 20–50k frames, mixed at ~30% per batch, with the ratio annealed down in the final epochs.
- Blender (full 3D neck with radius, real hand rigs) only if the 2.5D domain gap shows up in eval.

**Augmentation.**
- Photometric: color/white-balance jitter, gamma, noise.
- Capture artifacts: motion blur along random directions (60 fps strumming blur is real), JPEG/H.264 artifacts.
- Geometric: perspective jitter, rotation ±30°, scale 0.7–1.3, and h-flip at p ≤ 0.2. **H-flip swaps bass↔treble on
  every slot** to stay in the canonical right-handed orientation (D1). It does *not* simulate left-handed players; the
  app's mirror toggle handles them. Ablate h-flip on/off in M4.
- Occlusion: random hand-cutout pastes and cutout boxes.
- Crop: crop-jitter for Stage 2, so it tolerates tracker drift.

### 3.5 Auto-annotation pipeline (status 2026-10-11)

`src/gro/auto/` + `scripts/auto_annotate.py` produce `fretboard-kpts/v1` drafts (`source: pseudo` / `interp`,
`annotator: auto:gro-cv-v1`) without a trained model; every batch still goes through D8 review.

- **Classical CV, deterministic.** LSD segments that are bright thin ridges -> RANSAC neck axis over pairs of
  parallel, similar-length segments -> a strip sampled along the wires' vanishing point (wires exactly vertical) ->
  column ridge profile -> 1D projective fit of the 12th-root-of-2 positions (RANSAC over 3 peaks and small index
  gaps). Learnings: (1) a reversed geometric sequence is a projective image of itself (x -> 1/x), so the fit can't
  tell the nut direction; the prior "image spacing shrinks with the index" (physical ratio fret 1 : fret 24 = 3.8x)
  decides it. (2) Board-end edges and pickup edges pass every wire test; they're removed once numbering is known
  (index > `guitar.frets`) or when the AI names them.
- **AI only for targeted calls** (Vercel AI Gateway, `anthropic/claude-opus-5.5` for numbering,
  `claude-sonnet-5.5` for usability; cached in `.cache/ai/`): the fret number of the first labelled wire on a
  rectified strip (nut / inlays / last fret; ~$0.02-0.04 per frame), frame usability (~$0.003), and a rough
  fingerboard box when detection fails. Without a key the pipeline runs fully, with relative numbering
  (`index_unresolved`, never anchorable).
- **Temporal.** Keyframes every 1 s; KLT propagation + per-wire re-snap in between; the numbering is a session lock
  (later keyframes align to the propagated grid, no AI call); flags `fit_jump`, `wires_lost`, `keyframe_unaligned`,
  `keyframe_detect_failed`, `klt_lost`.
- **Eval** (`scripts/eval_auto.py`, 3 M1 GT frames + 2 extra hand-read frames in `samples/anchors_extra/`):
  before D13 (edge-convention GT): numbering 5/5 exact with AI; across-wire 0.65 px; treble 1.3 px; bass 5.6 px.
- **Strings, not edges (D13).** `gro.auto.strings`: the first (board-edge) fit gives a plane model. The board is
  resampled in plane coordinates, where strings are straight lines. Per fret cell, take the median across-neck ridge
  profile, soft-saturated so inlays don't dominate and the strongest string keeps a sharp peak. A line-score table
  S(c, d) sums these profiles over all cells. A 6-tooth comb (evenly spaced at every u, linear in u) is searched on
  S. A wire-presence term stops teeth from sitting where the wires have ended: the wire must continue on both sides
  of each string, so a tooth can't land off the board. The outer teeth are refined sub-sample. Anchors = snapped wire
  x outer string line; then refit. If the comb is weak (score <= 0.3), the strings fall back to the edges inset by
  8.5% (`EDGE_INSET`). Learnings: (1) when the strongest string is hard-clipped, the peak becomes a plateau and the
  argmax picks its edge (1.2 px bias), so use soft saturation. (2) With the high-E hidden, the comb slides one
  string outward unless teeth outside the wire ends are penalised. (3) The sampled v range must cover strongly
  diverging strings (tau≈0 board fits). (4) Refit tau on string anchors is 0.5–0.6, i.e. the string spread taper,
  not the board taper. Runtime is ~2.5x per frame (~13 s/frame on the eval frames, no AI).
- **Eval after D13** (string-convention GT; see D13 caveat): see the table in §3.5.1.

#### 3.5.1 Eval (2026-10-11, D13)

5 GT frames (220 anchors, 224 with AI-added nut anchors); string-convention GT. `eval/auto_eval_{ai,noai}.json`.

| run | mean px | median | max | bass mean | treble mean | across-wire | numbering exact | anchorable acc | cue acc |
|---|---|---|---|---|---|---|---|---|---|
| with AI | 0.88 | 0.53 | 16.0 | 0.72 | 1.04 | 0.70 | 5/5 | 4/5 | 3/5 |
| no AI | 0.75 | 0.51 | 7.13 | 0.67 | 0.84 | 0.58 | relative only | 1/5 | 1/5 |

With AI, the max (16 px) and most of the gap to no-AI come from the auto *nut* anchors on 2026-09-03_60s (the nut-step
finder; nut anchors only exist when the AI says the nut is visible). The rest are wires under fingers (vis 1, predicted
from the fit). The worst crops are in `review/eval-auto-worst-anchors/`. Anchorable/cue misses: 08-30_10s
(the GT says not anchorable, but the AI says the nut is visible) and 09-06_60s (cue 3-5-7 vs 9-12-15). Both predate D13.

#### 3.5.2 Cross-neck fix (2026-10-11)

Batch-002 review (Chase): wires + numbering mostly right, but 18/32 grids didn't span low E -> high E. Fix in
`gro.auto.strings` / `detect`: a **width prior** (outer-string span / local px per scale length / physical span
35.5 -> 52 mm is 0.80-1.15 on every GT and GOOD frame, 0.08-0.61 on the slivers) restricts the string comb, applied
for every index shift the relative numbering allows and exactly after numbering; short wires are grown to their
ridge ends; fallbacks comb -> edge inset (if in prior) -> prior span (flagged); a neck-coherence bonus against
overlay text; optional cached AI span check (flag only). `scripts/eval_auto.py` reports the cross-neck error next to
the fret-wire headline; 3 extra GT frames from the BAD set in `samples/anchors_crossneck/` (model-based outer strings,
~3-5% of span). Results and open issues: `eval/crossneck_report.md`.

## 4. Data collection

**Sources.** YouTube and TikTok guitar lessons and playthroughs with a close-up neck view: "scale lesson close up",
"fretboard cam", "lick tutorial", "neck cam", and non-English equivalents. Also Instagram reels via yt-dlp where
accessible.

**Criteria, all required:**
- At least ~8 consecutive frets visible.
- Neck ≥ 40% of frame width.
- ≥ 480p.
- Mostly static or slowly moving camera.
- A real fretboard (not a diagram).

**Diversity quotas** (track in the manifest; aim for coverage, not balance):

| Axis | Target mix |
|---|---|
| Guitar | **v1: ~100% electric (D3).** Later: ~55% electric, 30% steel acoustic, 15% classical/nylon |
| Inlays | dots, blocks, ornate (vine/tree/abalone), none/side-only — ≥15% ornate |
| Fretboard wood | dark (rosewood/ebony) and light (maple) — ≥20% maple |
| Handedness | No quota. Left-handed (and creator-mirrored) clips are welcome; they are flagged `mirrored` at ingest and become right-handed data (D1) |
| Lighting | bright studio, dim room, colored LEDs, backlit, mixed white balance |
| Camera | webcam on desk, phone on tripod, headstock-mounted/neck cam, over-shoulder (some) |
| Strings | 6-string primary; a few 7-string clips for robustness (labeled, not targeted) |

**Pipeline.**
- `tools/collect.py` runs on the Windows PC or another residential-IP machine, not the box (D10). It wraps
  `yt-dlp -f "bv*[height<=1080]"` and writes `data/manifest.csv` with these fields: `clip_id`,
  `source_url`, `uploader`, `license`, `download_date`, `duration`, `res`, `fps`, `mirrored`, tags.
- Segment downloads into 5–20 s clips with PySceneDetect.
- Prefilter clips with the M2 model (neck detected in >70% of frames). A human or agent then does a yes/no pass, which
  also sets `mirrored` per clip. Set it if the guitar looks left-handed, meaning the fretting hand is the player's right
  hand. This covers left-handed players and creators who mirrored their video. It is the only handedness judgment in
  the whole pipeline, and it's made once per clip, not per frame.

**Targets.**
- 200+ clips from 120+ distinct creators.
- 5–10k labeled frames: ~25–50 keyframes per clip, plus interpolated frames.
- 1k frames from the owner's own setup (§6).

**Licensing.**
- Internal model training only; approved by Chase (D6), and no commercial release is planned. Raw media is never
  committed, redistributed, or shown in public demos.
- Keep `source_url` and uploader for every clip, so we can honour takedowns by purging the clip and its derived frames
  and retraining.
- CC filtering is optional (D6). If a commercial release is ever planned, re-evaluate: the safest route is to retrain
  on owned, CC, and commissioned footage only. Track `license` per clip so that subset can be computed.

## 5. Training on the RTX 4090

**Environment.**
- Use WSL2 Ubuntu 24.04 on the Windows box: CUDA through the Windows driver, `uv` for Python 3.12, PyTorch 2.x
  (CUDA 12.x), and `ultralytics` (baseline).
- Use our own `fretnet` package for the custom model. Pin everything in `uv.lock`.
- Prefer WSL over native Windows. It avoids DataLoader worker issues and gives Linux parity with any cloud fallback.
- **Mac (runtime target):** `uv` + PyTorch (MPS) + `coremltools`, from the same `uv.lock` with platform markers. Use it
  for CoreML conversion checks, parity tests, `bench.py`, and the app. Training stays on the 4090.

**Access.**
- Run Windows OpenSSH with a WSL default shell, or `sshd` inside WSL on port 2222 with a port proxy.
- Enable Wake-on-LAN in BIOS and the NIC, and add `tools/wake.sh` (`wakeonlan <mac>`). This only works on the LAN or
  via a Tailscale subnet router, so put the box on Tailscale. Confirm with Chase that waking it remotely is acceptable.

**Data sync.**
- The archive is the 4090 PC's disk and/or Chase's Google Drive `Training data` folder (D9). The box keeps only
  working sets.
- `tools/sync.sh` runs `rsync -az --partial` over ssh/Tailscale for clips, frames, and labels.
- Labels and the manifest are small. Version them in git, or with DVC if they grow past ~50 MB.
- Pre-extract training frames to JPEG q95 (plus a crop cache) so the GPU isn't starved decoding H.264.

**Jobs.**
- `train.py --config configs/<exp>.yaml` runs inside `tmux`.
- Checkpoint every epoch to `runs/<exp>/`. Training auto-resumes, since the box may go off mid-run.
- A tiny `queue.sh` runs configs from `queue/` sequentially, so a batch can be queued before Chase shuts the lid on it.
- Expected runtimes:
  - YOLO11s-pose, 640px, 8k images: ~1–2 h for 150 epochs.
  - Custom net, 1024×256: ~2–4 h.

**Experiment tracking.**
- Weights & Biases on the free personal tier, logging configs, metrics, and overlay videos rendered on the eval clips
  each N epochs.
- Fallback: local TensorBoard plus a `runs/index.csv` row per experiment (git SHA, data version, metrics).

**Exports, in priority order (D2):**
1. **CoreML (v1 target).** Use `coremltools` from `torch.export`/TorchScript to an ML Program `.mlpackage` with fp16 and
   a fixed 1024×256 input. The YOLO baselines use ultralytics `format=coreml`. Check ANE/GPU residency on every export
   with the Xcode performance report or `MLComputePlan`, and benchmark `compute_units` options on the Mac.
2. **ONNX (opset 17) → TensorRT / ONNX Runtime CUDA**, for the 4090 prototype app.
3. **ONNX Runtime Web/WebGPU, and CoreML on iOS**, later.

Parity tests compare each export against PyTorch on 100 frames, with mean diff < 0.1 px after decoding. CoreML parity
runs on the Mac.

**Modal fallback.** Use it only if the box is unavailable for >1 week during a milestone. The same container image
(Dockerfile mirrors the uv env) runs on an A10G or L4.

## 6. Evaluation

**Splits are by creator/clip, never by frame.**

| Split | Contents |
|---|---|
| `train` / `val` | external clips (val = 10% of creators) |
| `test-ext` | 15% of creators, frozen at M2 |
| `test-owner` | Chase's 3 videos. **Fully held out through M3.** From M4, fine-tune on the first 60% of videos 1–2; video 3 + last 40% stay test forever |
| `ref` | 18 s reference screen recording: qualitative check, and a sanity test of the renderer (match the grid shape) |
| `test-lefty` | Left-handed clips from `test-ext` creators, evaluated **unmirrored** through the app's toggle path, with GT mapped back by `M`. This is the end-to-end check of D1 |

**Metrics.** These are computed on frames with ground truth and reported per split and per flag (occlusion, blur,
ornate inlay).

- **Normalized keypoint error.** Pixel distance divided by the local fret width at that keypoint (the distance between
  neighboring wires). Report mean, plus **PCK@0.1 and PCK@0.25 of fret width**, separately for visible and occluded
  points.
- **Fret-index accuracy.** % of frames with the correct index shift. Any off-by-one counts as a failure.
- **Cell IoU.** Render the 6×N cell polygons from the predicted vs ground-truth fit, for frets in the ground-truth
  visible range ±1. Report mean IoU and the % of cells with IoU > 0.5. This is the metric closest to what the user
  sees.
- **Jitter.** On segments where the ground-truth neck motion is < 0.5 px/frame (via KLT on GT), measure the RMS
  frame-to-frame displacement of the 4 filtered state corners, in px and in fret-width units. Also measure lag: the
  cross-correlation delay against GT on moving segments.
- **Latency.** p50 and p95 per stage and per target device, from the benchmark script `bench.py --device ...`.
- **Failure rate.** % of frames where the neck is present but there's no overlay, or cell IoU < 0.5, or the index is
  wrong. Also report time-to-reacquire after occlusion or exit.

**Acceptance targets for v1 (M4),** measured on `test-owner`:

| Metric | Target |
|---|---|
| PCK@0.1 (visible) | ≥ 90% |
| PCK@0.25 (occluded) | ≥ 80% |
| Index accuracy | ≥ 99.5% |
| Mean cell IoU | ≥ 0.80 |
| Jitter | ≤ 0.03 fret-width RMS |
| Failure rate | ≤ 2% |
| Compute on Chase's MacBook (CoreML) | ≤ 10 ms p95 |
| Mirror round-trip (mirrored input + toggle on vs original) | < 0.1 px |

## 7. Overlay renderer

### Music theory

- `tuning`: MIDI note numbers for strings 6→1. Standard is `[40,45,50,55,59,64]` (E2 A2 D3 G3 B3 E4). Presets cover
  Drop D, D standard, DADGAD, and Eb, plus custom.
- A note at `(string s, fret f)` is `pc = (tuning[s] + f) % 12`.
- A scale is `root_pc` plus an interval set. Initial scales:
  - Major, natural minor.
  - Major and minor pentatonic, blues.
  - The modes.
  - Harmonic minor, melodic minor.
  - Arpeggios / chord tones as "scales".
- A cell is in the scale if `(pc − root_pc) % 12 ∈ intervals`. Each cell is tagged with its degree, so the renderer can
  colour the root, 3rd, and 5th differently.
- Optional position or box filtering (e.g. CAGED shape 1, frets 5–8) for focus.
- Label notes as note names, degrees (1, b3, 5…), or none. Spell notes using the key signature (C# vs Db).

### Geometry

Everything is drawn in plane coordinates and projected through `H`, so perspective is free.

- **Fret wires** at `u_n`. **Strings** at `v_s = ±(1 − m)` interpolated across `s`, with an edge margin `m ≈ 0.08`.
  Strings converge toward the nut because `w(u)` tapers.
- **Cell `(s, n)`**: the quad between wires `n−1` and `n`, spanning midway between neighboring strings. Outer strings
  extend to the fretboard edge.
- **Note marker**: drawn at `u = u_{n−1} + 0.6·(u_n − u_{n−1})`, where a finger actually presses.
- **Inlay dots**: drawn at the standard positions, from a model of the guitar's inlays (single at 3, 5, 7, 9, 15, 17,
  19, 21; double at 12 and 24). They are drawn even when the real guitar has vines; that's a style choice that matches
  the reference.

### Styles

**"Reference" preset:**
- All cells in the visible range filled blue at α ≈ 0.35, with 1 px gaps so the wires show through.
- 1 px light-gray string lines at α 0.6.
- White inlay dots.
- Scale cells outlined 2 px in yellow; the root cell outlined in green.
- Out-of-scale cells left blue.

**"Dots" preset:** classic chart look, with filled circles on scale notes, a larger root, and no cell fill.

### Implementation

- Prototype: OpenCV `fillPoly` with alpha blending.
- When the left-handed toggle is on, the renderer projects with `M · H` (D1). Nothing else in the renderer changes,
  because strings and cells are addressed by physical identity.
- Product: a GPU pass (OpenGL in the Python app, and Metal if or when the app goes native on the Mac; WebGL/WebGPU
  later) drawing an instanced quad mesh. Vertices are computed in plane space and
  multiplied by `H` in the vertex shader. Render as 4–8 subdivided quads per cell so later non-planar radius
  corrections can bend them.
- Fade-in/out on track state.
- ~~Hand-occlusion mask.~~ Not needed (D5): drawing over the fingers is acceptable.

## 8. Milestones

| M | Deliverables | Exit criterion |
|---|---|---|
| **M0 — Foundations** (1 wk) | Repo layout (`fretnet/`, `tools/`, `configs/`, `app/`). Label schema v1 (with clip-level `mirrored`) + validators + YOLO/COCO converters (bass↔treble `flip_idx`). Frame extractor (applies `mirrored`). 4090 env + `sync.sh` + `wake.sh` + W&B. Mac env (`uv`, PyTorch MPS, `coremltools`). Owner videos inventoried on Drive. | `train.py` runs a smoke test on the 4090 from the Mac over ssh; a toy model converts to CoreML and runs on the Mac |
| **M1 — Geometry & renderer on GT** (1–2 wk) | §1.2 fit library with unit tests (synthetic `H`, index-shift recovery, mirror composition `M·H`, h-flip swap keeps the orientation). Annotation tool v1 (keyframes, fit, snap, interp, zoom). 300 frames of owner video hand-labeled (**test only**). Renderer v1 ("reference" style) playing on GT fits. Scale/tuning engine with tests. | Overlay on GT labels visually matches the reference look; fit RMSE < 1.5 px on labeled frames |
| **M2 — Data + baseline** (2–3 wk) | 200+ clips collected with manifest. Agent annotation protocol running + gold set (200 frames). 3k labeled frames. YOLO11s-pose baseline + eval harness (§6), all metrics, overlay videos. Stage 1 OBB detector. **First live prototype on the 4090 PC + webcam** (baseline via PyTorch CUDA/ONNX). | Baseline numbers on `test-ext`/`test-owner`; failure modes catalogued; live overlay seen on a real webcam |
| **M3 — Real model** (2–3 wk) | Rotated-crop heatmap net with aux heads + spacing loss. Robust fit + One Euro + KLT propagation + gating/hysteresis. Synthetic 2.5D generator v1. **CoreML export smoke test on the Mac**: fixed shapes, residency report, first `bench.py` numbers, so ANE-hostile ops are caught before M4. Live prototype loop on the 4090 (TensorRT). | Beats baseline on PCK and cell IoU; index accuracy ≥ 98%; Stage 2 converts to CoreML with no CPU-fallback layers |
| **M4 — Scale up & hit targets** (2–3 wk) | Pseudo-labeling rounds, 8–10k labeled frames, synthetic mix tuned, fine-tune on owner train split. Ablations (crop vs full frame, aux heads, spacing loss, synthetic ratio, h-flip on/off). | §6 accuracy targets met on `test-owner` |
| **M5 — Mac real-time app** (2 wk) | Python desktop app on the MacBook: webcam capture → CoreML (ANE or GPU, chosen by bench) → GPU render. Same code runs on the 4090 with a TensorRT backend. **Left-handed / mirrored-camera toggle (D1).** UI for key/scale/tuning/style/fret range. `bench.py` latency report on M-series. Go native (Swift + AVFoundation + Metal) only if Python can't hold the latency target. | 60 fps live on Chase's MacBook (camera permitting) with ≤ 10 ms p95 compute; mirror round-trip < 0.1 px and `test-lefty` within 2% of `test-ext` on cell IoU; usable in a real practice session |
| **M6 — Polish, then other platforms** (open) | Acoustic/classical support (D3). Per-guitar calibration (τ, fret count, inlay map), auto-saved. 7-string and capo robustness. Later (D2): port to web/WebGPU and/or iOS/CoreML. | Defined after M5 and the remaining §10 answers |

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Fret-index off-by-one** (nut off-frame, ornate inlays) | Overlay shifted a whole fret. Worst possible UX. Now a hard requirement to solve, since nut-less framing must work (D4) | Inlay aux head + body-joint/last-fret landmarks (spacing alone is shift-invariant, §1.2; metric-shape cue only as a tie-breaker), hysteresis, per-session lock, zero-tolerance QA. PoC may require the nut |
| **Fretting-hand occlusion** exactly at the region of interest | Missing or wrong frets where the user looks | Occluded-but-inferable labels (vis = 1), parametric fit fills gaps, hand-paste augmentation |
| **Ornate (vine) inlays** confused with wires or markers | False peaks, index errors | Line-map aux head, ornate inlays oversampled (≥15%), masked inlay loss, owner guitars in M4 fine-tune |
| **Thin wires near the body** at webcam resolution | Large normalized error at high frets | Rotated crop from full-res frame, stride-4 heatmaps, report metrics per fret range |
| **Domain gap** YouTube → home webcam (exposure, noise, compression) | Test-owner much worse than test-ext | Owner fine-tune split, webcam noise/compression augmentation, collect more owner-like clips |
| **Agent annotation quality** | Noisy labels, silent index errors | Snap + fit + overlay verification, gold set gating, agents only `draft` |
| **4090 availability** | Stalled training | Queue + auto-resume, WoL/Tailscale, Modal as a last resort with the same image |
| **Licensing of scraped video** | Low for now: internal use approved, no commercial release planned (D6) | Per-clip license tracking kept so a retrain-on-owned path stays possible |
| **Planar assumption** (radius, neck relief, strings above the board) | Few-px errors on outer strings | Measure in cell IoU; add per-string `v` / height offset only if needed |
| **Fast neck motion / motion blur** | Lag or lost track | One Euro β tuning, KLT coast, blur augmentation, 60 fps capture |
| **CoreML/ANE op coverage** (layers falling back to the CPU, fp16 drift) | Misses the Mac latency target, or parity fails | CoreML smoke test from M3, crop sampling kept outside the graph, fixed shapes, per-export residency check, GPU compute units as a fallback |
| **Wrong mirror toggle** (or an unexpectedly mirrored camera feed) | The model sees off-orientation input, and the overlay is poor or missing | Toggle labeled "Left-handed / mirrored camera"; after repeated track loss the app shows a hint to check the toggle. This is a hint only, with no handedness detection (D1) |
| **Mac camera frame rate** (built-in MacBook cameras are typically 30 fps) | 60 fps target unreachable on the built-in camera | Confirm the camera (§10 Q14); an external 60 fps webcam, or accept 30 fps with a 30 Hz model |

## 10. Open questions for Chase

1. ~~**Target platform for v1.**~~ **Answered 2026-10-11 → D2.** Desktop webcam on an Apple Silicon MacBook (CoreML,
   ANE/MPS), with the 4090 PC for early prototypes; web and iOS later.
2. **Camera placement.** Is the home-office webcam angle in your videos *the* canonical setup, or should we support a
   phone on a tripod, neck cams, or over-the-shoulder? Narrower scope means a faster v1.
3. ~~**Guitars.**~~ **Answered 2026-10-11 → D3.** Electric only for now; all types eventually. *Still open:* fret
   counts of your guitars (22 vs 24), and any 7-strings?
4. ~~**Left-handed support.**~~ **Answered 2026-10-11 → D1.** Required, as a UI mirror toggle; never inferred from
   video.
5. **Scales and display.** Which scales and modes matter first? Note names, degrees, or plain dots? Is the reference's
   blue-cell look the target, or the "dots" style? Should a root or "target note" be user-selectable per exercise?
6. **Tunings.** Standard only, or Drop D / Eb / open tunings? Capo support?
7. ~~**Fret range.**~~ **Answered 2026-10-11 → D4.** Every fret visible in the camera, robust to partial necks and a
   hidden nut (the first PoC may require the nut).
8. ~~**4090 logistics.**~~ **Answered 2026-10-11 → D7.** Usable any time, including overnight.
9. ~~**Data policy.**~~ **Answered 2026-10-11 → D6.** Downloading YouTube/TikTok clips for internal training is OK; no
   commercial release planned.
10. **Annotation budget.** How many hours can you personally spend on the gold set (~200 frames ≈ 3–4 h with the tool)
    and on QA review?
11. ~~**Hand occlusion.**~~ **Answered 2026-10-11 → D5.** Drawing over the fingers is fine.
12. **Reference video.** May we use the 18 s screen recording frames to tune the renderer style (colors, alpha,
    proportions)?
13. **Which MacBook exactly?** Which chip (e.g. M3/M4 Pro or Max), how much RAM, and which macOS version? This sets
    the ANE-vs-GPU choice and which CoreML features we can use (e.g. `MLComputePlan` needs macOS 14.4+).
14. **Which webcam on the Mac?** Built-in MacBook cameras are typically 30 fps. Is there an external 60 fps webcam,
    or is 30 fps acceptable for v1? Where does it sit relative to the guitar (this ties into Q2)?
15. **Your own footage looks left-handed.** In all three owner videos the headstock is on the image left and the
    "Ibanez" logo and background text read correctly, so the camera isn't mirrored: these are left-handed guitars
    played left-handed. Per D1 these clips get `mirrored: true`, and the M1 demo exercises that path. Please confirm
    (and confirm the live app will run with the toggle on by default for you).
16. **Bass-edge convention for labels.** Where should the bass (low-E) board edge go when the board edge itself
    isn't visible: (a) at the low-E string line, which is what the current GT does, (b) where the fret wire ends on
    the face, or (c) wherever the outermost string/edge highlight runs? The auto pipeline disagrees with GT by ~5 px
    along the wire only on this side (treble 1.3 px, wire position 0.65 px). One convention for both GT and auto would
    make the numbers comparable (§3.5, `review/eval-auto-worst-anchors/`).
    *Provisionally answered by D13 (anchors on the outer strings). Please check `review/d13-gt-strings/`: dashed
    lines are the detected strings, red the outer two, green circles the old anchors. Confirm or overrule.*
