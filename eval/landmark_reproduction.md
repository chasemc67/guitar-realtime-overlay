# Landmark numbering experiment

Stage 1 is `runs/fretnet/v4q_r50_832/last.pt`. The new stage is
`src/gro/fretnet/landmark.py`: torchvision MobileNetV3 Small ImageNet initialization,
height-pooled convolutional features, seven 1D localization heads at quarter-fret
spacing, and one explicit absent class per landmark. Landmark order is 12, 5, 7, 9,
3, 15, nut (0). Fret 12 receives twice the training loss weight.

The model reuses the v7 whole-neck rectification (1024×128, fitted fret coordinates
−2..26, transverse coordinates −0.35..1.35). Cached extended GT strips are shared
with v7. Positions are GT-fit wire centers, not manually marked inlay centers.
Presence requires at least one labelled wire endpoint inside the original image;
unlabelled wires are absent targets. This assumes the existing labels' visibility
is complete. Inference receives only the stage-1 fit and RGB image.

Training uses the full scale-v1aug train/val splits; the shared filename guard
rejects all configured eval_v2 IDs and 09-06 dates, including augmented copies.
Continuous ±3-fret crop shifts, projective corner jitter, color changes, blur and
occlusion simulate stage-1 errors. Targets undergo the identical projective jitter.
Random augmentation is seeded by epoch and sample. Validation uses fixed seeded
shifts, no photometric jitter, and selects the checkpoint by minimum validation loss.
Training plans 12 epochs, AdamW 2e-4, cosine decay, batch 4, BF16, two CPU threads.

Inference confidence is posterior mass within ±0.5 fret of each peak, including
competition with the absent class. The fixed threshold is 0.8, chosen before
held-out evaluation. Each confident landmark snaps to the nearest observed stage-1
wire within 0.75 fret. Its vote is landmark number minus that wire's slot. Votes
are summed by confidence; a strict majority wins, and ties/no evidence fall back
to stage 1. Pixel coordinates and sides are preserved. With `--wire-reassign`,
snapping uses fitted integer indices and the existing geometric reassignment
provides the fallback. `--wire-reject-ambiguous` uses the existing 0.45-fret rule;
that rule was evaluation-informed in v7 and is not reselected here.

Fusion adds one confidence-weighted vote from the existing validation-selected
v7 MobileNet 1536 offset classifier. Its confidence also must meet 0.8. Three
agreeing landmarks can outweigh one conflicting landmark and the offset vote.
If no landmark is confident, a confident offset classifier can still correct
numbering; otherwise the stage-1 fallback applies. Landmark-only and fusion
share the same landmark checkpoint and thresholds.

Run in the PC's `~/gro/repo` using its existing `.venv`:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_landmark.py \
  --name v8_landmark_1024 --width 1024 --epochs 12 --batch 4

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_v2.py \
  --weights runs/fretnet/v4q_r50_832/last.pt \
  --landmark runs/fretnet/v8_landmark_1024/best.pt \
  --device cuda --name v8_landmark
```

For `v8_fusion`, add `--numbering runs/fretnet/v7num_1536/best.pt`.
For the corresponding `_geometry` variants, add
`--wire-reassign --wire-reject-ambiguous`. All four use stage-1 letterbox pixels.
The optional existing `--numbering-source-resolution` is also supported.
Python inference uses `gro.infer.load_backend(stage1_path, landmark=checkpoint,
numbering=optional_offset_checkpoint, threshold=0.8)`.

`eval_v2.py` retains the frozen macro D13/numbering/coverage/detection metrics and
09-06 honest-set metrics. The new `landmark_diagnostics` report is pooled by
landmark, separately for actual stage-1 strips and evaluation-only GT strips.
It records labelled presence counts, presence accuracy (posterior ≥0.5), detection
counts, localization precision/recall (confidence ≥0.8 and error ≤0.5 fret), and
mean position error among confident detections of present landmarks. Mislocalized
detections count against precision and recall. Stage-1 fit failures count as
missed detections. Landmarks outside the predicted strip count as absent there.
GT-strip diagnostics are computed only after final predictions, never used for votes.

Checkpoint paths remain on the PC; no binary weights are added to the repository.
No existing v6 job was signalled, and this session makes no git commits.

Validation of implementation: PC full suite passes all 100 tests, including eight
landmark tests; local suite passes 74 with three torch-dependent modules skipped.
Ruff passes on all changed Python files. Tests exercise conflicting and tied votes,
missing observed wires, absence competition, positive/negative offset signs,
coordinate-preserving corrections, failed-fit fallback, checkpoint round-trip,
GT target jitter, missing-label targets, detection accounting, and transformation
of original-frame GT into letterbox-fit strip coordinates.

All 12 epochs completed in about 6.3 minutes of epoch time. Validation selected
epoch 2 (loss 2.006876, localization-or-absence accuracy 0.811368). The first
detached launch exited before training; the successful run was kept alive through
an open SSH session. The four evaluations ran after epoch 12 completed, using
`best.pt`. All four produced the complete 232+15-frame reports. Training/runtime,
initialization and checkpoint hashes, split hashes, and all epoch metrics are in
[`v2/v8_landmark_training.json`](v2/v8_landmark_training.json). Results and the
per-landmark table are in [`model_permissive.md`](model_permissive.md).
