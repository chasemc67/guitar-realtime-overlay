# Board-end clipping, per-clip mirrored flag, jitter (model v1, demo v2), 2026-10-11

Renders (not in git): `/workspace/guitar-overlay/demo/v2/demo_v2_{0906_60s,stan,nBpe2}.mp4` (H.264, 960 wide;
6.4 / 10.3 / 4.6 MB). Contact sheets: `eval/endclip_v2/*_sheet.jpg` (15 spread frames + 2 runs of 5 consecutive frames).

## 1. Board end (last fret)

`gro.board_end`: per frame, a probe fit on slots <= 21 (so slots the model may place on the pickups can't pull the
fit), then the high-fret strip is rectified through it. Each predicted wire position 19..24 is tested for a two-sided
ridge (brighter than the wood a quarter cell away on both sides, in both the bass and treble halves of the strip),
normalized by the median ridge of wires 14..18. A wire counts when the ratio is >= 0.35, or >= 0.175 with keypoint conf
> 0.5. The frame's estimate is the last consecutive supported wire, or none if the next wire is out of view or the
count is below 20. `FretCountVote` is a running mode with hysteresis (24 until 15 votes, then switch only on a 10-vote
lead). `--frets N` (or the manifest `frets` column) overrides. Grid, strings and notes stop at the last fret, and a
white board-end line is drawn 0.4 cell past it.

| Clip | Detected | Per-frame votes (demo run) | Switched from default 24 at | Manual wire count |
|---|---:|---|---|---|
| 09-06 Ibanez | **24** | 24: 3409, 23: 98, 22: 84, 21: 3, 20: 2 | (never switched) | 24 (`endclip_v2/end_0906_1430.jpg`) |
| Stan Telecaster | **22** | 22: 1950, 21: 217, 20: 134 | frame 14 (0.47 s) | 22 (`end_stan_1150.jpg`: dots in cells 17/19/21, then one more wire before the board end) |
| nBpe2 Strat | **22** | 22: 747, 21: 111, 20: 34, 23: 15, 24: 2 | frame 20 (0.67 s) | 22 (`end_nBpe2_400.jpg`) |

Keypoint confidence alone can't do this: median conf of slot 24 on the 24-fret Ibanez is 0.22 (23: 0.51), while slot 22
on the 22-fret guitars is 0.71–0.79. Low per-frame votes come from the fretting hand covering wires 20..22.

## 2. Mirrored flag

`data_manifests/*.tsv` now have `mirrored` (yes/no) and `frets` (blank = detect) columns. `owner_videos.tsv` covers
Chase's videos and `data/test_ext/stan_june2`. `gro.auto.manifest.clip_info()` is used by demo_model.py,
cache_preds.py and eval_model.py, with `--[no-]mirrored` to override. See PLAN D1.

## 3. Jitter (scripts/jitter_eval.py on the prediction caches; frets 24/22/22)

Wire midpoints (y = 0.5) for wires 1..last, in source px and in % of the local fret width:
step = median frame-to-frame motion on static-ish frames; hf = median residual after a centered 0.2 s moving average;
lag = median distance from the centered average of the raw fit on fast frames (> 5% fret width per frame).

| Clip (source) | Variant | step px (% fret) | hf px (% fret) | lag px (% fret) |
|---|---|---:|---:|---:|
| 09-06 (720p60) | raw per-frame fit | 0.12 (0.53%) | 0.45 (1.76%) | 1.01 (6.3%) |
| | v1 anchor EMA a=0.4 | 0.22 (0.85%) | 0.32 (1.25%) | 1.48 (8.7%) |
| | **v2 One-Euro** | 0.15 (0.59%) | 0.22 (0.88%) | 1.11 (6.8%) |
| Stan (1080p30) | raw | 0.91 (2.16%) | 1.16 (2.98%) | 1.97 (5.6%) |
| | v1 EMA | 0.60 (1.45%) | 0.63 (1.63%) | 3.55 (9.8%) |
| | **v2 One-Euro** | 0.26 (0.66%) | 0.49 (1.25%) | 2.61 (7.3%) |
| nBpe2 (1080p30) | raw | 1.53 (3.88%) | 1.84 (4.78%) | 2.54 (6.8%) |
| | v1 EMA | 0.76 (1.98%) | 0.93 (2.44%) | 4.32 (11.0%) |
| | **v2 One-Euro** | 0.40 (1.02%) | 0.88 (2.21%) | 2.72 (7.3%) |

v2 (`gro.smooth.OneEuroModel`): One-Euro filtering of the image positions of control points (wires 0/6/12/18/last ×
outer strings), then a DLT refit of H. One shared cutoff = 0.4 Hz + 3 × max(0, speed − 2 fret widths/s), with speed
measured on the in-view points. The noise floor matters: without it, detector noise reads as motion and opens the
filter. Grid searched: min_cutoff {0.2..1.5}, beta {0.1..3}, floor {0, 0.5, 1, 2}. v2 beats v1 on every number for
every clip, with about 25–35% less lag on the 30 fps clips. Raw 09-06 step (0.53%) is already lower than both smoothed
variants. That metric only counts static frames, and at 60 fps a smoother's slow drift shows up there.

## Still wrong / not done (from looking at the frames)

- The fret count is voted from 24 down, so the first 0.5–0.7 s of the 22-fret clips draw 24 frets.
- The overlay draws over the fretting hand (no occlusion handling), as in v1.
- On the yellow maple Tele, the blue cells barely show; the style isn't adapted to board colour.
- On Chase's 09-06 the nut sits under the fret wrap. Numbering there is still the model's absolute slots (see
  `model_v1_0906_honest.md`).
- Jitter uses each variant's own track as reference (no GT trajectories). Lag is relative to a non-causal average of the
  raw fit, so it understates true lag where the raw fit itself is biased.
- The board-end check assumes bright wires on a contrasting board. It isn't tested on dark wires / light boards beyond
  the Tele, or on scalloped boards.
