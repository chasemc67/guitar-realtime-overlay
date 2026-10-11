# Cross-neck fix: eval report (2026-10-11)

Brief: `_briefs/cross_neck_fix.md`. Batch-002 (Chase-judged): 14 GOOD / 18 BAD. The wires and numbering were mostly right,
but the grid often didn't span low E -> high E.

## What changed (src/gro/auto)
- **Width prior.** rho = outer-string span / (px per scale length along the neck) / physical span (35.5 mm at the
  nut -> 52 mm at the 24th, 648 mm scale). It is 0.80-1.15 on every GT and GOOD frame and 0.08-0.61 on the slivers.
  The string comb only searches spans with rho in [0.7, 1.35] at both ends of the visible run and a taper within
  +-25% (no drift). The prior is applied for every index shift the relative numbering allows, and exactly once the
  numbering is known (the pipeline refits with the shift).
- **Slope range bug.** When the first board model was a sliver, the comb's slope range in plane units couldn't
  express the physical taper, so every in-prior comb was rejected. The range now scales with the prior span.
- **Wire growth.** Wires shorter than 1.2x the local fret spacing (the neck hypothesis only covered a binding or a
  single string) are grown to their ridge ends and re-snapped.
- **Fallbacks.** comb -> edge inset (only if within the prior) -> prior span centred on the wires. The fallbacks are
  flagged `strings_edge_inset` / `strings_prior`, and rho is stored in `auto.strings_rho`.
- **Text/watermark.** Hypotheses whose wires sit under a six-string comb get a quality bonus, and 3 hypotheses get the
  full wire stage. nbkiBItGzoU no longer locks onto the "licklibrary.com" text, but it still doesn't find the real
  neck (it now lands on the amp grille, flagged `strings_prior` + `ai_span_bad`).
- **AI span verifier** (`claude-sonnet-5.5` via the Gateway, cached, flag only): "does the grid cover all 6 strings?" on
  a rendered crop -> `ai_span_bad`. About $0.005 per frame.
- **Acoustic s7DVJX4PK3U** is excluded via the manifest `exclude` column; auto_annotate skips it.
- **Relative-numbering trim** now limits the index span, not just the wire count (one frame was numbered to fret 25).

## Eval (`scripts/eval_auto.py`, with AI)
GT = the 5 D13 frames (`samples/anchors{,_extra}`) + 3 new cross-neck frames from the BAD set
(`samples/anchors_crossneck`, made with `scripts/gt_crossneck.py`, crops in `review/gt-crossneck/`).
"Before" = commit bd7419f (pre-fix pipeline) with the new eval script. Pooled values are anchor-weighted means.
JSON: `eval/crossneck_before_ai.json`, `eval/crossneck_after_ai.json`.

| GT set | run | frames | fret-wire px (perp) | anchor px | span err px | span err % | outer-string px | outer-string % | numbering exact | wire recall |
|---|---|---|---|---|---|---|---|---|---|---|
| D13 GT (5 Chase frames) | before | 5 | 0.70 | 0.88 | 0.5 | 0.7 | 0.3 | 0.5 | 5/5 | 100% |
| D13 GT (5 Chase frames) | after | 5 | 0.71 | 0.86 | 0.5 | 0.7 | 0.3 | 0.4 | 5/5 | 100% |
| cross-neck GT (3 BAD yt frames) | before | 3 | 0.68 | 34.87 | 67.2 | 73.1 | 34.8 | 38.5 | 3/3 | 100% |
| cross-neck GT (3 BAD yt frames) | after | 3 | 1.76 | 17.57 | 12.6 | 14.3 | 17.2 | 17.2 | 3/3 | 85% |
| all 8 | before | 8 | 0.69 | 12.74 | 23.7 | 26.0 | 12.3 | 13.7 | 8/8 | 100% |
| all 8 | after | 8 | 1.03 | 5.95 | 4.2 | 4.8 | 5.4 | 5.5 | 8/8 | 94% |

| frame | before: span % / outer % / perp px / numbering | after: span % / outer % / perp px / numbering | after: strings source |
|---|---|---|---|
| 2026-08-30_15-06-39_10s#0 | 1.0 / 0.9 / 0.72 / ok | 2.5 / 1.4 / 0.82 / ok | comb |
| 2026-09-03_20-38-02_10s#0 | 1.5 / 0.7 / 0.44 / ok | 0.4 / 0.3 / 0.47 / ok | comb |
| 2026-09-06_09-00-01_10s#0 | 0.2 / 0.1 / 0.66 / ok | 0.8 / 0.4 / 0.64 / ok | comb |
| 2026-09-03_20-38-02_60s#0 | 0.9 / 0.7 / 1.19 / ok | 0.6 / 0.3 / 1.13 / ok | comb |
| 2026-09-06_09-00-01_60s#0 | 0.2 / 0.1 / 0.51 / ok | 0.1 / 0.2 / 0.51 / ok | comb |
| I4Vt5Q4aUrw_08#0 | 70.8 / 35.4 / 0.51 / ok | 6.4 / 4.1 / 1.33 / ok | comb |
| IZJdHyfkrsg_08#0 | 48.3 / 23.9 / 0.37 / ok | 37.5 / 18.7 / 1.57 / ok | prior |
| mP2wtBlcCSc_08#0 | 89.8 / 49.8 / 1.01 / ok | 2.8 / 35.2 / 2.59 / ok | prior |

Columns: fret-wire px = anchor distance perpendicular to the GT wire (the D13 headline). Span err = |pred - GT|
low-E..high-E distance per wire. Outer-string = distance of each predicted anchor from the GT outer-string line.

**Reading it**
- **D13 GT:** unchanged (fret-wire 0.70 -> 0.71 px, cross-neck < 1% of span). No regression on Chase's footage.
- **BAD frames:** span error 73% -> 14% of span, outer-string error 38% -> 17%.
  - I4Vt5Q4aUrw is fixed (6% / 4%, comb).
  - mP2wtBlcCSc has the right width (2.8%) but is shifted ~35% of a span towards the bass side (prior fallback,
    centre from the wires). Its wire recall also dropped to 58%.
  - IZJdHyfkrsg is still 37% off in span (prior fallback, `fit_rmse_high`).
- **Regression:** on the BAD frames the fret-wire error rose from 0.68 to 1.76 px. Wire growth / re-snapping moves
  some wires, and the new GT's wires were seeded from the old auto wires, so "before" is biased low there. This needs
  a look before the growth step is trusted.

**GT caveats.** Only 3 of the planned 6-10 frames are labelled. Outer strings in this footage are too dim/blurred to
read to the pixel, so the GT strings are hand-read board edges inset by the physical edge->E margin (or read directly
where visible, I4Vt5Q4aUrw): good to ~3-5% of span, enough to score coverage. The wires come from the old auto
labels, re-snapped, with numbering checked on the inlays/nut.
- SGTxeiB_jBM was left out: lens curvature (board-edge local slope -0.30 at the body end vs a -0.37 chord). A
  homography can't fit it, which likely explains the "drift" Chase saw.
- QYoi / HiSa / 3FoeH were not finished in the time box.

## Batch-002 re-run (`review/batch-002-auto-yt-v2/`, labels `labels/auto/batch-002-auto-yt-v2/`)
31 frames (acoustic skipped). rho on the numbered anchors (0.75-1.25 = width plausible):

| set | before in range | after in range |
|---|---|---|
| GOOD (13) | 13 | 13 |
| BAD (18) | 0 | 17 |

Strings source after: comb 21, prior 8 (all flagged), edge_inset 2. `ai_span_bad` flags 6 frames: 8-grcb, NN224,
QYoi, hAuP, mP2w, nbki.

My visual read of the v2 sheet, at contact-sheet resolution: about 11 of the 18 BAD frames now look full-width.
- **Look full-width:** 3FoeH, Av9D, Beb0, I4Vt, HiSa, QYoi, Rztm, aZH5, tWLg, wi9Z, xQg.
- **Better but unsure:** SGTx, IZJd.
- **Still wrong:** mP2w (shifted), nbki (wrong region), NN224 (oblique), hAuP (small), 8-grcb (partial).

These need Chase's judgement (same procedure as `judged.csv`). In-range rho shows the width is plausible, not that
the position is right.

## Open issues
1. The prior fallback centres on the wires' midpoints. That is often off by a string or two (mP2w): use the comb's
   best in-prior position even at low score, or ask the AI for the side.
2. The wire-growth step raises fret-wire error on the BAD frames (0.68 -> 1.76 px); gate it on the prior instead of
   applying it whenever wires are short.
3. Lens distortion (SGTxeiB) needs per-clip undistortion or a curved model.
4. Watermark frames: the real neck in nbkiBItGzoU is never a top hypothesis. This needs a neck-region detector
   (or AI rescue when no hypothesis has strings).
5. Finish the GT (QYoi, HiSa, 3FoeH + oblique NN224 once numbering resolves).
6. Runtime is ~1-3 min per frame under load (comb search with wide prior ranges). The 32-frame batch was sharded 4 ways.
