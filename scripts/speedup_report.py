"""Render collected speed experiments and append newly landed results to the status log."""

# ruff: noqa: E501 -- rendered Markdown table rows
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/speedup"
STATUS = Path("/workspace/jam-dashboard/_briefs/permissive_r2_status.md")
START = "<!-- speedup-results-start -->"
END = "<!-- speedup-results-end -->"


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def table(records, mode="CPU_AND_NE"):
    lines = [
        "| Candidate | CPU_AND_NE median / p95 ms | ALL median / p95 ms | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in records:
        if mode not in rec.get("held", {}):
            continue
        m = rec["held"][mode]["macro"]
        h = rec["held"][mode]["honest_0906"]
        times = [rec["latency"].get(u, {}) for u in ["CPU_AND_NE", "ALL"]]
        t = [f"{f(t.get('median_ms'), 2)} / {f(t.get('p95_ms'), 2)}" for t in times]
        lines.append(
            f"| {name} | {t[0]} | {t[1]} | {f(m['d13_mean_px'], 2)} / {f(m['d13_median_px'], 2)} | "
            f"{f(m['numbering'], 6)} | {f(m['coverage'], 6)} | {f(h['d13_mean_px'])} / {f(h['numbering'])} / {f(h['coverage'], 6)} |"
        )
    return "\n".join(lines)


def main():
    records = []
    for p in sorted(OUT.glob("ml_held_*.json")):
        rec = json.loads(p.read_text())
        records.append((p.stem.removeprefix("ml_held_"), rec))
    preamble = """# Speeding up v4q without losing numbering accuracy

No Jam Dashboard app changes, release actions, or git commits. Experiments use the
PC RTX 4090 and the Mac Mini M4. Full protocol: [plan.json](speedup/plan.json).
This report is regenerated as results land; missing candidates are still pending.

Targets: CPU_AND_NE median approximately 8 ms or less, eval_v2 numbering >0.601,
and 09-06 numbering 1.000. Historical v1: 4.25 ms, numbering 0.601 / 1.000;
v4q-832 + 0.30 refit: 28.05 / 28.69 ms, numbering 0.714223 / 1.000,
eval_v2 D13 5.2753 / 2.1632 px, coverage 0.846059;
09-06 D13 0.7621 px, coverage 0.987216. These are prior measurements.

## Compression correctness finding

Initial default `r832_int8`, `r832_palette6`, and `r832_palette4` variants compressed
all eligible constants. A graph audit found this included `gx`/`gy` coordinate grids
and positional constants, not only learned weights. Those results are retained.
The distinct `r832_weights_*` variants and subsequent student/fine-tune compression
restrict compression to convolution/linear weights, preserving geometric constants.
This is an implementation correction based on the exported graph; thresholds and
bit widths are not retuned against held-out scores. See
[graph audit](speedup/coordinate_compression_audit.json). Later runs perform VAL
parity before held-out scoring. Compression reports record the exact scope.

## Measured CoreML results

**Accuracy below comes from actual Mini CoreML CPU_AND_NE outputs**, including
compressed candidates, followed by the frozen refit. Timings show both compute
modes: 20 warmups / 200 calls, batch 1, the existing Python CoreML backend
(including PIL conversion, excluding resize/refit/video decoding). No concurrent
experiment is timed on the Mini. Detection and numbering denominators are in the
per-frame JSON reports; parity uses 32 VAL calibration images, never held-out selection.
Raw CoreML outputs are retained as `.CPU_AND_NE.npz` and `.ALL.npz` beside the reports.

"""
    lines = [preamble, table(records)]
    passed = []
    for name, r in records:
        if "CPU_AND_NE" not in r.get("held", {}):
            continue
        h = r["held"]["CPU_AND_NE"]
        if (h["macro"]["numbering"] or 0) > 0.601 and h["honest_0906"]["numbering"] == 1.0:
            passed.append((name, r))
    lines += ["", "## Recommendations", ""]
    strict = [(n, r) for n, r in passed if r["latency"]["CPU_AND_NE"]["median_ms"] <= 8.0]
    if strict:
        name, r = min(strict, key=lambda v: v[1]["latency"]["CPU_AND_NE"]["median_ms"])
        lines.append(
            f"**{name} meets all three strict targets**: {r['latency']['CPU_AND_NE']['median_ms']:.2f} ms; "
            f"eval_v2 numbering {r['held']['CPU_AND_NE']['macro']['numbering']:.6f}; 09-06 1.000."
        )
    else:
        lines.append("No collected CoreML candidate yet meets all three strict targets (≤8.00 ms, >0.601, 1.000).")
    if passed:
        name, r = min(passed, key=lambda v: v[1]["latency"]["CPU_AND_NE"]["median_ms"])
        lines.append(
            f"Fastest numbering-qualified candidate so far: **{name}**, "
            f"{r['latency']['CPU_AND_NE']['median_ms']:.2f} ms. No model is promoted by this experiment."
        )
    lines += [
        "",
        "## PyTorch held-out results",
        "",
        "These are separate runtime variants, scored once after VAL selection. Do not treat their accuracy as CoreML accuracy.",
        "",
        "| Candidate | eval_v2 D13 mean / median px | Numbering | Coverage | 09-06 D13 / numbering / coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in sorted(OUT.glob("held_*.json")):
        r = json.loads(p.read_text())
        m = r["macro"]
        h = r["honest_0906"]
        lines.append(
            f"| {p.stem.removeprefix('held_')} | {f(m['d13_mean_px'])} / {f(m['d13_median_px'])} | "
            f"{f(m['numbering'], 6)} | {f(m['coverage'], 6)} | {f(h['d13_mean_px'])} / {f(h['numbering'])} / {f(h['coverage'], 6)} |"
        )
    lines += [
        "",
        "## VAL selection",
        "",
        "All refit sweeps use 1,498 VAL images and thresholds 0.30–0.60; choose numbering, then coverage, then lower D13. "
        "Every checkpoint is selected on VAL mean keypoint error. Filename guards exclude eval_v2 IDs and 09-06 from train/VAL. "
        "640 was selected for distillation from the resize VAL sweep before held-out scoring. "
        "All four students use eight epochs, GT + teacher heatmap/keypoint/query-kernel/confidence losses; "
        "fine-tunes use four epochs and GT losses. Separate extended_512 and extended_640 continuations select the backbone using VAL numbering only, then each independently train 20 additional epochs from the same 640 student checkpoint, with the same 832 teacher. Compression candidates use VAL numbering plus latency only (see plan).",
        "",
        "| Candidate | Selected threshold | VAL numbering | VAL coverage | VAL D13 mean px |",
        "|---|---:|---:|---:|---:|",
    ]
    selections = list(OUT.glob("resize_*.json")) + list(OUT.glob("selection_*.json"))
    for p in sorted(selections):
        s = json.loads(p.read_text())
        v = s["val"][str(s["chosen_threshold"])]
        lines.append(
            f"| {p.stem.removeprefix('selection_')} | {s['chosen_threshold']:.2f} | {f(v['numbering'], 6)} | {f(v['coverage'], 6)} | {f(v['d13_mean_px'])} |"
        )
    lines += [
        "",
        "## Tracking",
        "",
        "Each sparse labelled target is paired with its actual preceding decoded video frame. "
        "Inference/refit on that predecessor is propagated by LK optical flow and a RANSAC homography. "
        "This reports **skipped-frame-only accuracy**, not accuracy of a densely labelled full video. "
        "Effective adjacent-pair time is (inference + preprocess + Python refit + tracking) / 2, excluding decode. "
        "It includes Python postprocessing overhead and is not a native app FPS claim. "
        "Window 21/31 is selected on VAL synthetic translations before held-out frames. The original 832 control uses native-image flow; subsequent optimized variants also select long edge 320/640/native, choosing the fastest VAL configuration within a fixed 1-source-pixel mean-error budget. Four eval_v2 targets at frame zero have no predecessor and are excluded, leaving 228 eval_v2 and all 15 09-06 targets (explicit IDs in JSON). The first baseline attempt stopped at the first such target before scoring any held-out frame; the corrected run reused its frozen VAL selection.",
        "",
        "| Candidate | Effective pair median / p95 ms per frame | Flow median / p95 ms | eval_v2 D13 mean / median px | Numbering / coverage | 09-06 D13 / numbering / coverage | Flow failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(OUT.glob("ml_tracking_*.json")):
        if ".selection." in p.name:
            continue
        r = json.loads(p.read_text())
        m = r["macro"]
        h = r["honest_0906"]
        t = r["effective_pair_per_frame_ms"]
        flow = r["flow_ms"]
        lines.append(
            f"| {p.stem.removeprefix('ml_tracking_')} | {f(t['median'])} / {f(t['p95'])} | {f(flow['median'])} / {f(flow['p95'])} | "
            f"{f(m['d13_mean_px'])} / {f(m['d13_median_px'])} | {f(m['numbering'], 6)} / {f(m['coverage'], 6)} | "
            f"{f(h['d13_mean_px'])} / {f(h['numbering'])} / {f(h['coverage'], 6)} | {r['failures']} |"
        )
    lines += [
        "",
        "Contiguous-video replay uses alternating inference/flow on up to 120 frames each "
        "from 09-06, nBpe2 and Stan. There is no dense ground truth; model-to-model agreement in "
        "the raw reports is **not** an accuracy measurement.",
        "",
        "| Candidate / clip | Frames | Actual alternating pair median / p95 ms per frame | Flow failures |",
        "|---|---:|---:|---:|",
    ]
    for p in sorted(OUT.glob("ml_video_*.json")):
        r = json.loads(p.read_text())
        for c in r["clips"]:
            lines.append(
                f"| {p.stem.removeprefix('ml_video_')} / {Path(c['path']).stem} | {c['frames']} | "
                f"{f(c['pair_median_ms'])} / {f(c['pair_p95_ms'])} | {c['flow_failures']} |"
            )
    lines += [
        "",
        "## ALL compute-mode accuracy",
        "",
        table(records, "ALL"),
        "",
        "## Reproduction and limits",
        "",
        "- Scripts: `scripts/speedup_{train,eval,bundle,export,compress,coreml,tracking,video}.py`; "
        "`speedup_collect.sh` transfers frozen checkpoints, and `speedup_finalize_mini.py` chooses compression candidates without held-out scores.",
        "- PC weights: `~/gro/repo/runs/fretnet/speedup_*/best.pt`; Mini exports: `~/gro-coreml/speedup/`. "
        "Per-checkpoint protocols/logs and selection hashes are under `eval/speedup/`; raw images and large checkpoints stay on the machines.",
        "- CoreML compression uses coremltools 9.0: symmetric int8 linear weights, k-means palettes, "
        "and activation calibration using 32 deterministic VAL-only frames. "
        "[Apple documentation](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) "
        "explains why compression speedups must be measured on the target hardware.",
        "- All new backbones/weights are torchvision BSD-3-Clause with the existing ImageNet caveat; "
        "see `LICENSES.md`. No Ultralytics code/weights enter this work.",
        "- PC full test suite: 104 passed (including backbone checkpoint/export and synthetic homography tests). "
        "New Python scripts pass Ruff and compile checks; collection shell passes `bash -n`.",
        "- Short training budgets are not convergence claims. Validation uses pipeline labels; "
        "eval_v2 contains pseudo-GT plus hand corrections, while 09-06 uses hand GT.",
        "",
    ]
    lines += [
        "",
        "## CoreML parity against matching PyTorch checkpoints (VAL, 32 frames)",
        "",
        "Coordinates are source-image pixels. Fit disagreement counts either runtime lacking a fit; "
        "it is not necessarily a fit lost by CoreML. Raw and refitted predictions are compared separately.",
        "",
        "| Candidate / units | Raw keypoint mean / p95 px | Refit keypoint mean / p95 px | Refit wire mean / p95 px | Refit anchor flips/frame | Fit disagreements |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in records:
        for units, parity in record.get("parity", {}).items():
            raw, refined = parity["raw"], parity["refit"]
            lines.append(
                f"| {name} / {units} | {f(raw['kpt_px_mean'])} / {f(raw['kpt_px_p95'])} | "
                f"{f(refined['kpt_px_mean'])} / {f(refined['kpt_px_p95'])} | "
                f"{f(refined['wire_px_mean'])} / {f(refined['wire_px_p95'])} | "
                f"{f(refined['anchor_flips_per_frame'])} | {refined['fit_lost_frames']} |"
            )
    lines += [
        "",
        "## Direct-inference controls on the same tracking targets",
        "",
        "Aggregated from saved direct CoreML predictions on exactly the 228 + 15 tracking targets. "
        "No additional model evaluation or tuning is performed.",
        "",
        "| Tracking variant / direct reference | Direct eval_v2 D13 mean / median | Direct numbering / coverage | Direct 09-06 D13 / numbering / coverage |",
        "|---|---:|---:|---:|",
    ]
    for path in sorted(OUT.glob("ml_control_*.json")):
        control = json.loads(path.read_text())
        m, h = control["macro"], control["honest_0906"]
        lines.append(
            f"| {control['tracking']} / {control['reference']} | {f(m['d13_mean_px'])} / {f(m['d13_median_px'])} | "
            f"{f(m['numbering'], 6)} / {f(m['coverage'], 6)} | {f(h['d13_mean_px'])} / {f(h['numbering'])} / {f(h['coverage'], 6)} |"
        )
    (ROOT / "eval/speedup.md").write_text("\n".join(lines))
    model = ROOT / "eval/model_permissive.md"
    text = model.read_text()
    block = (
        START
        + "\n\n### Measured speedup CoreML rows\n\n"
        + table(records)
        + "\n\nAccuracy from CPU_AND_NE CoreML outputs; "
    )
    block += "[full protocol, ALL accuracy, tracking and recommendations](speedup.md).\n\n"
    block += "\n".join(lines[lines.index("## PyTorch held-out results") : lines.index("## VAL selection")]) + "\n" + END
    if START in text:
        text = text[: text.index(START)] + block + text[text.index(END) + len(END) :]
    else:
        text += "\n\n" + block + "\n"
    model.write_text(text)
    seenpath = OUT / "reported.json"
    seen = json.loads(seenpath.read_text()) if seenpath.exists() else []
    for name, r in records:
        if name in seen or "ALL" not in r.get("held", {}):
            continue
        m = r["held"]["CPU_AND_NE"]["macro"]
        h = r["held"]["CPU_AND_NE"]["honest_0906"]
        t = r["latency"]["CPU_AND_NE"]
        line = f"{datetime.now(timezone.utc).isoformat(timespec='minutes')} speedup CoreML {name}: CPU_AND_NE "
        line += (
            f"{t['median_ms']:.2f}/{t['p95_ms']:.2f} ms; eval_v2 D13 {f(m['d13_mean_px'])}/{f(m['d13_median_px'])}, "
        )
        line += f"numbering {f(m['numbering'], 6)}, coverage {f(m['coverage'], 6)}; 09-06 {f(h['d13_mean_px'])}/{f(h['numbering'])}/{f(h['coverage'], 6)}.\n"
        with STATUS.open("a") as dest:
            dest.write(line)
        seen.append(name)
    for path in sorted(OUT.glob("held_*.json")):
        name = "torch:" + path.stem.removeprefix("held_")
        if name in seen:
            continue
        result = json.loads(path.read_text())
        m, h = result["macro"], result["honest_0906"]
        line = f"{datetime.now(timezone.utc).isoformat(timespec='minutes')} speedup {name}: "
        line += f"eval_v2 D13 {f(m['d13_mean_px'])}/{f(m['d13_median_px'])}, numbering {f(m['numbering'], 6)}, "
        line += (
            f"coverage {f(m['coverage'], 6)}; 09-06 {f(h['d13_mean_px'])}/{f(h['numbering'])}/{f(h['coverage'], 6)}.\n"
        )
        with STATUS.open("a") as destination:
            destination.write(line)
        seen.append(name)
    seenpath.write_text(json.dumps(seen, indent=2))


if __name__ == "__main__":
    main()
