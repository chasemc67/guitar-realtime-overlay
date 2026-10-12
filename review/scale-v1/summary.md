# scale-v1: label + filter summary

Settings: keyframes every 1.0 s, propagation to 5.0 fps. 97 segments, 4269 s of video, 16.6 worker-hours (14.0 worker-s per video-s). AI $13.01 (numbering cached 0, numbering calls 274, rescue cached 0, rescue calls 3, span cached 0, span calls 800, usability cached 0, usability calls 90).

Filter: rho in [0.75, 1.25], rmse <= 1.5 px, >= 6 visible wires, AI span verdict of the nearest checked keyframe within 3.0 s.

## trainval

| | count |
|---|---|
| videos | 89 |
| segments | 91 |
| frames labelled | 19828 |
| keyframes | 3040 |
| kept | 6136 (31%) |
| kept + anchorable (absolute numbering) | 5566 |
| kept, NOT anchorable (relative only; excluded from absolute training) | 570 |
| videos with >= 1 kept frame | 58 |
| dropped | 13692 |

| drop reason | first reason | any reason |
|---|---|---|
| no_label | 3803 | 3803 |
| ai_unusable | 2692 | 3527 |
| ai_span_bad | 3787 | 4929 |
| strings_prior | 2559 | 6069 |
| rho_out | 312 | 1267 |
| rmse_high | 356 | 1782 |
| few_wires | 124 | 519 |
| track_bad | 59 | 581 |

Anchor cues (kept): inlay_9_12_15 4555, NONE 570, nut 534, inlay_3_5_7 477

## test (2026-09-06, TEST ONLY: pseudo-GT for eval, never trained on)

| | count |
|---|---|
| videos | 1 |
| segments | 6 |
| frames labelled | 1257 |
| keyframes | 251 |
| kept | 1247 (99%) |
| kept + anchorable (absolute numbering) | 0 |
| kept, NOT anchorable (relative only; excluded from absolute training) | 1247 |
| videos with >= 1 kept frame | 1 |
| dropped | 10 |

| drop reason | first reason | any reason |
|---|---|---|
| no_label | 0 | 0 |
| ai_unusable | 0 | 0 |
| ai_span_bad | 0 | 0 |
| strings_prior | 5 | 5 |
| rho_out | 5 | 5 |
| rmse_high | 0 | 0 |
| few_wires | 0 | 0 |
| track_bad | 0 | 0 |

Anchor cues (kept): NONE 1247

