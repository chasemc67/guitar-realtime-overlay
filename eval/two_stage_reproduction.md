# Two-stage numbering reproduction

Run from the repository root on the PC, in its existing `.venv`. No Ultralytics code or
weights are used. The ResNet-18 and MobileNetV3 Small backbones use torchvision ImageNet
initialisation; see `LICENSES.md` for provenance and dataset caveats.

## Geometry controls

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_v2.py \
  --weights runs/fretnet/v4q_r50_832/last.pt --device cpu \
  --wire-reassign --name v7_geometry_control

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_v2.py \
  --weights runs/fretnet/v4q_r50_832/last.pt --device cpu \
  --wire-reassign --wire-reject-ambiguous --name v7_geometry_reject
```

Both keep observed pixel coordinates. The second drops points at least 0.45 fret from
the nearest integer wire in the homography instead of retaining their original slot.
This policy was added after inspecting the first control's failing 09-06 frame; its
result is an evaluation-informed development result, not a fresh independent test.
Its eval_v2 numbering is 0.66454, D13 5.3703 / 2.2148 px, coverage 0.86963, detection
1.000; 09-06 numbering 1.000, D13 0.7693 px, coverage 0.98864.

## Learned models

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_numbering.py \
  --name v7num_1024 --arch resnet18 --width 1024 --batch 4 --epochs 12 --workers 4

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_numbering.py \
  --name v7num_1536 --arch mobilenet_v3_small --width 1536 --batch 4 --epochs 12 --workers 4

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_numbering.py \
  --name v7seq_1024 --arch mobilenet_v3_small --mode sequence \
  --width 1024 --batch 2 --epochs 8 --workers 4
```

The smaller backbone for subsequent variants limits contention with the existing v6
jobs. This changes both width and backbone in the second variant; it is not an isolated
width ablation. The ResNet run completed 10 of 12 planned epochs, then was stopped to release GPU memory
after more than ten minutes without another completed epoch. Its validation-selected
checkpoint is epoch 9 (accuracy 0.890139); no held-out scores were used for this choice.
The first run started before the `--arch` option was introduced and uses
ResNet-18; the checkpoint loader defaults legacy v7 checkpoints to ResNet-18.
The `seed=17` checkpoint field is the torch initialisation seed; augmentation uses fresh
NumPy generators and is not bitwise reproducible. Reports contain all per-frame metrics.

The sequence head samples 25 tokens from the whole-neck CNN features, mixes their
context with two transformer encoder layers, and predicts an absolute index at every
fitted wire. Each point retains its original stage-1 number when its sequence decision
is below 0.8 confidence. Checkpoints use validation accuracy (per-wire for sequence,
per-frame offset for the classifier), never eval_v2 or 09-06 labels, for selection.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_v2.py \
  --weights runs/fretnet/v4q_r50_832/last.pt --device cpu \
  --numbering runs/fretnet/v7num_1024/best.pt --name v7num_1024
```

Substitute the other numbering checkpoint paths to evaluate them. Add `--wire-reassign`
to make geometric slot reassignment the stage-1 fallback and apply confident classifier offsets on top, and
`--wire-reject-ambiguous` for the explicit rejection policy. Those two optional flags
apply to offset models; the sequence model has its own per-wire assignment and fallback.
All runs use the same frozen `configs/eval_v2.json` and report GT-strip and predicted-strip
offset accuracy separately from the all-wires-correct numbering metric.

The geometry controls retain predictions in all 232 eval_v2 frames. One Jackson V frame
becomes unassessable for numbering under the frozen nearest-wire criterion; counting it
as a failure leaves both macro numbering values unchanged. All 15 honest frames remain
assessable. `eval/v2/v7_environment.json` records runtime versions and hashes of the
stage-1 weights, frozen config, and training/validation filenames plus labels.
