# v4q speed experiments

In progress, 2026-10-13 UTC. No application/release changes or git commits.
Target: Mini CPU_AND_NE model median approximately 8 ms or less, eval_v2 numbering
above 0.601, and 09-06 numbering 1.000. Baseline v4q-832 + VAL-selected refit:
28.05 / 28.69 ms, eval_v2 D13 5.2753 / 2.1632 px, numbering 0.714223,
coverage 0.846059; 09-06 D13 0.7621 px, numbering 1.000, coverage 0.987216.

All new refit thresholds are selected using the 1,498-image scale-v1aug VAL split
before held-out evaluation. Checkpoints use train/VAL only; filename guards exclude
all configured eval_v2 IDs and 09-06. Held-out scores never choose sizes, epochs,
thresholds or training parameters. Each frozen candidate is evaluated once per
runtime/compute mode. Mini timings use 20 warmups and 200 calls, batch 1.

| Inference-only size | VAL numbering | VAL threshold | Mini CPU_AND_NE median / p95 ms | Mini ALL median / p95 ms |
|---|---:|---:|---:|---:|
| 640 | 0.750334 | 0.30 | 14.110 / 14.732 | 17.086 / 17.466 |
| 512 | 0.456914 | 0.30 | 8.103 / 8.260 | 10.305 / 10.534 |
| 416 | 0.357143 | 0.30 | 5.493 / 5.571 | 8.214 / 8.520 |

640 was chosen for all four distilled students from VAL accuracy, before looking
at held-out results. Eight epochs per student are budgeted, checkpoint chosen by
minimum VAL mean keypoint error. GT is combined with teacher heatmap, keypoint,
query-kernel and confidence targets. The teacher sees the same augmented image
at 832; the student sees its resized 640 version. Compatible query/FPN/head tensors
initialize from the teacher; backbone initialization is from torchvision.
Fine-tuning the teacher at 640/512/416 is separately budgeted for four epochs each.

Weight compression: int8 linear weights and k-means 6-/4-bit palettes; activation
quantization will be attempted using 32 deterministic VAL-only calibration frames.
The [CoreML optimization documentation](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
recommends measuring device-specific latency rather than assuming weight compression
accelerates inference. Compressed accuracy will be scored from Mini CoreML outputs.

Tracking uses true previous-frame/current-frame pairs. Since held-out labels are
sparse, every labelled target is treated as an intermediate frame; the preceding
frame provides the inference/refit. Report skipped-frame-only accuracy and measured
pair cost divided by two, including preprocessing/refit/flow, excluding video decode.
This is a one-step adjacent-pair proxy, not a continuously labelled video benchmark.
The LK window is selected on VAL images under synthetic known translations first.

Raw evidence and frozen selections: `eval/speedup/`. Scripts: `speedup_eval.py`,
`speedup_train.py`, `speedup_bundle.py`, `speedup_coreml.py`, `speedup_compress.py`,
and `speedup_tracking.py` in `scripts/`.
