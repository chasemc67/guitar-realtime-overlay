# Model v1 on Chase's held-out 2026-09-06: honest visual eval + fret-wrap check

Weights: `runs/pose/v1/weights/best.pt` (yolo11s-pose, imgsz 640), run on the PC. Inference uses the **canonical
(h-flipped) frame**, as for the labels and training (D1: the clip is flagged `mirrored: true`). The bad first demo
came from running `demo_model.py` without `--mirrored`. That was a usage error, not a model failure: on the flipped
frame the model lines up (below).

## 1. New hand GT: 15 frames spread over 09-06

`samples/anchors_0906/2026-09-06_09-00-01_{5,20,35,50,70,85,100,115,130,145,160,175,195,215,240}s.json`. The frames
are in `samples/frames/`, built with `scripts/hand_anchors_0906.py`, and the proof sheets are in `review/gt-0906/`.

- **Method.** I read zoomed, grid-labelled crops (`scripts/zoom.py`) of the canonical frame and clicked three wires per
  frame:
  - fret 1, the first wire on the body side of the black fret wrap. The nut is under the wrap.
  - fret 2.
  - fret 24, the last wire before the end of the board. The board-end edge itself is not a wire: its gap is larger
    than the 24→23 spacing.
- **Fill-in.** Every other wire was filled in by growing outwards from the clicked wires one at a time. Each step fits
  a least-squares 12-TET model, predicts the next wire, and snaps it onto its ridge (`gro.snap.snap_wire`). A final
  pass re-snaps every wire on the full fit.
- **Occlusion.** Wires with a weak ridge (under a finger) keep the model line and get vis 1. That happened to 0–1
  wire per frame.
- **D13 anchors.** Anchors are the wire lines intersected with the outer-string lines found by the string comb
  (`detect_strings`), the same convention as `samples/anchors*`.
- **Numbering check.** All 15 frames have 24 wires with a 12-TET fit RMSE of 0.27–0.63 px. An off-by-one at either
  end cannot fit both the fret-1/2 spacing and the fret-24 end, so the low RMSE is itself a numbering check. I checked
  every proof sheet by eye:
  - each GT wire sits on a wire;
  - fret 1 is next to the wrap;
  - fret 24 is at the board end;
  - the 12th-fret inlay block falls between wires 11 and 12.
- **Caveat.** Positions come from ridge snapping, which is also what the auto labels use. Both therefore mark the
  ridge centre, which slightly favours label-trained models in px error. The *numbering* is fully independent of the
  pipeline.

## 2. Fret-wrap check: is the index shifted?

`scripts/wrap_check.py` matches every owner-video hand GT frame to the nearest scale-v1 auto label (≤5 frames away).
Each labelled keypoint is assigned to the GT wire it lies on, and the per-frame shift is the most common (label index −
GT index). The output is `eval/wrap_check.json`.

| Labels vs hand GT | Frames | Index shift | Nearest-wire px (median of per-frame medians) |
|---|---:|---|---:|
| 09-06 (test-only video; 15 new + 10 s + 60 s) | 17 | **0 in 17/17** (100 % of matched keypoints each) | 0.3 (one 5.7 px frame: 60 s, 4 frames of motion away) |
| 08-30 (wrap; training video) | 1 | 0 | 0.6 |
| 09-03 (no wrap, nut visible; training video) | 2 | 0 | 0.2–0.4 |

| Model v1 predictions vs hand GT | Frames | Index shift | D13 slot median px | Nearest-wire median px |
|---|---:|---|---:|---:|
| 09-06 held-out | 17 | **0 in 17/17** | 0.3–1.3 per frame; pooled mean 0.86 / median 0.64 | identical (shift 0) |
| 08-30 / 09-03 (training videos) | 3 | 0 in 3/3 | 0.66 / 1.12 / 7.17 | (see `model_v1_0906_honest_report.md`) |

**Conclusion: there is no index shift.** Neither the labels nor the model read the wrap as the nut, and neither
mislabels the first visible wire. The details are less reassuring than the headline numbers, though:

- **The 09-06 labels are right by luck of convention, not because the wrap was understood.**
  - In `frames.csv`, all 1,247 kept 09-06 frames have `numbering_resolved=0` and no anchor cue. The pipeline fell back
    to `relative_numbering` (first detected wire = fret 1, D11).
  - On this guitar the wrap covers only the nut gap, so the first visible wire really is fret 1.
  - On a frame where the fretting hand hides fret 1, or a wrap that covers fret 1, the same fallback would shift
    everything by −1. 7 kept frames already start at fret 2, carried over by tracking.
- **08-30 has 7 kept frames with cue `nut`.** On the hand-GT frame, the label's nut keypoint is at fret position
  −0.09/−0.10. That position comes from the 12-TET fit to the GT wires, where 0 is the true nut. So the "nut" is the
  real nut or the locking-nut clamp just behind it, not the wrap's edge. A wrap edge would sit at about +0.3…+0.7.
  Only one GT frame backs this up.
- **The model puts its nut slot at the true nut position, under the wrap.** Fret position is −0.12…+0.03 on all 17
  held-out frames. Its confidence there is only about 0.09, so the nut never passes the 0.5 threshold. The nut also
  stays low-confidence on 09-03, where it is visible (0.05). The model has not learned to *see* a nut. It extrapolates
  where the nut should be, which on wrap footage is the right behaviour.

## 3. Visual alignment (scripts/eval_model.py `--align`)

New mode:

```
python scripts/eval_model.py --weights runs/pose/v1/weights/best.pt --name v1_0906_honest --hand-only --no-timing \
    --align review/eval-0906 --gt samples/anchors --gt samples/anchors_extra --gt samples/anchors_0906
```

For every held-out hand frame it renders GT wires (green, index above the low-E string) against predicted wires and
keypoints (magenta, index below the high-E string, conf > 0.5) on the canonical frame. Renders are in
`review/eval-0906/`. Per frame it reports:

- D13 slot px: the error against the GT wire with the same index;
- nearest-wire px: the error against the closest GT wire, ignoring its index;
- the index shift.

The full tables are in `eval/model_v1_0906_honest_report.md`.

| Held-out hand GT (09-06) | Frames | D13 mean / median px | Outer-string mean px | Coverage (conf>0.5) | Detection | Numbering | Index shift ≠ 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1, canonical frame | 17 | **0.86 / 0.64** | 1.44 | 0.947 | 17/17 | 17/17 | 0/17 |

**Do they line up?** Yes. I opened the renders (5 s, 60 s, 70 s, 240 s, among others):

- Predicted wires sit on the GT wires from fret 1 to fret 22/23, including under the fretting hand (fingers over
  frets 6–10).
- The labels match: magenta 1 is next to the wrap and the 12 pair brackets the 12th-fret inlay.
- The worst frame is 60 s (D13 median 1.3 px). There the predicted high-E line runs about 3 px inside the GT high-E
  line over frets 1–9, which is an outer-string offset, not a wire offset.
- 85 s also has a 3.5 px outer-string mean.

What does **not** line up:

1. **Fret 24 is never output.** Its confidence is 0.17–0.27 on every frame, but the raw (below-threshold) slot-24
   keypoints are close to the GT wire: D13 median 1.57 px, max 3.9 px. 23 is borderline (0.42–0.59): it drops out on
   4/17 frames. This is the entire coverage gap
   (0.947) and the reason "index set exact" is 0.
2. **The nut slot is never output**, even when the nut is visible (09-03). This is fine for the overlay, which
   extrapolates fret 0 from the fit, but it means the model can't be the source of a nut cue.
3. Only 17 still frames. Temporal stability and blur were not measured; the fixed demo render `demo_v1b` on the PC
   covers that.

The earlier v1 report numbers for the 09-06 pseudo-GT (D13 median 0.81 px, numbering 0.968) are consistent with
this. They no longer depend on unresolved pseudo-GT numbering, because the hand GT confirms it.

## 4. Proposal: fret-wrap-aware handling

1. **Wrap detector, and the nut cue disabled when a wrap is present.**
   - Detection rule: a dark, matte band across the whole board width within about 0.3–1.2 fret-1 spacings on the
     headstock side of the first wire. It can be checked on the fitted model by sampling luminance and texture across
     the band, the same way `_nut_step` profiles the nut.
   - When it fires:
     - never accept an AI or profile "nut visible" answer, and never add nut anchors;
     - flag the frame `fret_wrap`;
     - put the nut at the 12-TET extrapolation (what the model already does).
   - Cheap AI fallback: on keyframes, ask "is there a fret wrap/dampener near the nut?" once per segment and cache the
     answer.
2. **Inlay-first / end-of-board numbering instead of the relative fallback.** On wrap footage, resolve the absolute
   index from cues that survive the wrap:
   - the 12th-fret double/large inlay;
   - the 3-5-7-9 inlay rhythm;
   - fret 24 / board end. On a 24-fret neck, the last wire before a board-end edge is fret 24, and the 12-TET fit then
     pins fret 1.

   Add `board_end` as a D12 anchor cue next to `nut`, `inlay_9_12_15` and `inlay_3_5_7`. Keep the relative fallback
   for frames with none of these, but mark them non-anchorable, as now. On 09-06, 1,145 of 1,247 kept frames reach fret 24, so most of
   them would become resolved and anchorable.
3. **Synthetic wrap augmentation in training.** On labelled frames whose nut is visible (YouTube, 09-03), paint a
   dark band (black/grey/patterned, 0.3–1.0 fret-1 widths, slight perspective, random texture and specular
   highlights) between the nut and fret 1, with p ≈ 0.2. The keypoints stay unchanged: nut v=1 (occluded but
   inferable). This teaches the model to extrapolate the nut rather than snap it to the wrap edge, and gives the
   detector training negatives.
4. **Model-side fixes from this eval.**
   - Per-slot confidence calibration or threshold for slots 0 and 24. Most training frames have v=0 there, so their
     confidence is depressed.
   - Alternatively, emit the full grid from the fitted 12-TET model and use per-slot confidence only for visibility
     and occlusion. The demo already fits geometry.
   - Add a wrap-occluded fret 1 test frame (hand over frets 1–2) to the hand GT. That is the case where the relative
     fallback would break.
