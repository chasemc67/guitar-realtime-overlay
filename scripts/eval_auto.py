"""Evaluate the auto-annotation pipeline against hand-labelled ground truth.

    uv run python scripts/eval_auto.py                    # GT = samples/anchors (+ samples/anchors_extra if present)
    uv run python scripts/eval_auto.py --no-ai            # deterministic only (relative numbering)
    uv run python scripts/eval_auto.py --gt DIR --gt DIR2 --json eval/auto_eval.json

Metrics (per frame and pooled):
- px error per GT anchor (only GT anchors with vis > 0; also vis 2 only), mean / median / max, split bass / treble,
  and the error perpendicular to the GT wire (independent of where the board edge is put along the wire).
  Predicted and GT wires are matched geometrically first (best index offset d), so a numbering error does not turn
  into a pixel error; recall = GT anchors whose wire was found.
- cross-neck (does the grid span low E -> high E): per matched wire, the span width error |pred span - GT span|
  and the outer-string error (distance of each predicted anchor from the GT low-E / high-E line, a line fitted
  through the GT anchors of that side), in px and in % of the GT span at that wire. The fret-wire (perpendicular)
  error stays the headline (D13); this is reported next to it.
- numbering: exact fret-index match (d == 0), reported for every frame and for GT-anchorable frames (D12).
- anchorable flag and anchor_cue accuracy vs the GT labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.ai import AIClient
from gro.auto.pipeline import AutoConfig, annotate_frame
from gro.snap import fit_line

ROOT = Path(__file__).resolve().parents[1]


def _wire_lines(anchors):
    by = {}
    for a in anchors:
        by.setdefault(a.fret, {})[a.side] = np.array([a.x, a.y])
    return {f: (d["bass"], d["treble"]) for f, d in by.items() if "bass" in d and "treble" in d}


def compare(gt: list[A.Anchor], pred: list[A.Anchor]) -> dict:
    gt = [a for a in gt if a.vis > 0]
    gl = _wire_lines(gt)
    pl = _wire_lines(pred)
    pm = {f: 0.5 * (b + t) for f, (b, t) in pl.items()}
    best = None
    for d in range(-30, 31):
        hits = 0
        for f, (b, t) in gl.items():
            if f + d in pm:
                u = (t - b) / np.linalg.norm(t - b)
                n = np.array([-u[1], u[0]])
                hits += abs((pm[f + d] - b) @ n) < 4.0
        if best is None or hits > best[0] or (hits == best[0] and abs(d) < abs(best[1])):
            best = (hits, d)
    _, d = best
    pa = {(a.fret, a.side): a for a in pred}
    rows = []
    for a in gt:
        p = pa.get((a.fret + d, a.side))
        if p is None:
            rows.append((a, None, None))
            continue
        b, t = gl[a.fret]
        u = (t - b) / np.linalg.norm(t - b)
        n = np.array([-u[1], u[0]])
        e = float(np.hypot(p.x - a.x, p.y - a.y))
        perp = float(abs((np.array([p.x, p.y]) - b) @ n))
        rows.append((a, e, perp))
    return {"offset": d, "rows": rows}


def crossneck(gt: list[A.Anchor], pred: list[A.Anchor], offset: int) -> dict:
    """Span width error and outer-string error per matched wire (px and % of the GT span)."""
    gt = [a for a in gt if a.vis > 0]
    gl, pl = _wire_lines(gt), _wire_lines(pred)
    side_lines = {}
    for i, side in enumerate(("bass", "treble")):
        pts = np.array([gl[f][i] for f in sorted(gl)])
        if len(pts) >= 2:
            side_lines[side] = fit_line(pts)
    out = {"span_px": [], "span_pct": [], "outer_px": [], "outer_pct": [], "span_signed_pct": []}
    for f, (b, t) in gl.items():
        if f + offset not in pl:
            continue
        pb, pt = pl[f + offset]
        span_gt = float(np.linalg.norm(t - b))
        e = float(np.linalg.norm(pt - pb)) - span_gt
        out["span_px"].append(abs(e))
        out["span_pct"].append(100 * abs(e) / span_gt)
        out["span_signed_pct"].append(100 * e / span_gt)
        for side, p in (("bass", pb), ("treble", pt)):
            if side not in side_lines:
                continue
            c, u = side_lines[side]
            d = float(abs((p - c) @ np.array([-u[1], u[0]])))
            out["outer_px"].append(d)
            out["outer_pct"].append(100 * d / span_gt)
    return out


def worst_crops(canon, rows, offset, pred, k: int = 6, half: int = 18, zoom: int = 6):
    """Zoom crops (canonical orientation) around the k worst GT anchors: GT = green circle, auto = red cross."""
    pa = {(p.fret, p.side): p for p in pred}
    worst = sorted((r for r in rows if r[1] is not None), key=lambda r: -r[1])[:k]
    tiles = []
    for g, e, _ in worst:
        p = pa[(g.fret + offset, g.side)]
        cx, cy = int(round(0.5 * (g.x + p.x))), int(round(0.5 * (g.y + p.y)))
        x0, y0 = cx - half, cy - half
        crop = cv2.copyMakeBorder(canon, half, half, half, half, cv2.BORDER_CONSTANT)[
            y0 + half : y0 + 3 * half, x0 + half : x0 + 3 * half
        ]
        t = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)

        def P(q, x0=x0, y0=y0):
            return int((q.x - x0 + 0.5) * zoom), int((q.y - y0 + 0.5) * zoom)

        cv2.circle(t, P(g), 9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(t, P(p), (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(t, f"{g.fret}{g.side[0]} {e:.1f}px", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(t, f"{g.fret}{g.side[0]} {e:.1f}px", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(t)
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    return np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0))) for t in tiles])


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max())}


def load_gt(dirs: list[Path]):
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            lf = A.load(p)
            for i, fr in enumerate(lf.frames):
                if fr.neck_present and fr.present():
                    yield p, lf, i, fr


def frame_image(lf, fr, label_path: Path):
    ff = fr.extra.get("frame_file")
    for base in (label_path.parent, ROOT):
        if ff and (base / ff).exists():
            return cv2.imread(str(base / ff))
    vp = ROOT / lf.video.get("path", "")
    if vp.exists():
        cap = cv2.VideoCapture(str(vp))
        if "frame_index" in fr.extra:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr.extra["frame_index"]))
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(fr.t) * 1000)
        ok, img = cap.read()
        cap.release()
        return img if ok else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, action="append", help="GT label dirs (default samples/anchors[+_extra])")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--offline", action="store_true", help="AI answers from the cache only")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--crops", type=Path, default=None, help="write GT-vs-auto zoom crops of the worst anchors here")
    a = ap.parse_args()
    dirs = a.gt or [d for d in (ROOT / "samples/anchors", ROOT / "samples/anchors_extra") if d.exists()]
    ai = None if a.no_ai else AIClient(**({"model": a.model} if a.model else {}), offline=a.offline)
    if ai is not None and not ai.available:
        print("note: no AI_GATEWAY_API_KEY -> running without AI", file=sys.stderr)
    cfg = AutoConfig(use_ai=not a.no_ai, ai_usability=False, ai_span_check=False)
    cn_keys = ("span_px", "span_pct", "outer_px", "outer_pct")
    results, pooled = [], {k: [] for k in ("all", "vis2", "bass", "treble", "perp", *cn_keys)}
    for path, lf, i, fr in load_gt(dirs):
        img = frame_image(lf, fr, path)
        if img is None:
            print(f"skip {path.name}#{i}: no image", file=sys.stderr)
            continue
        canon = cv2.flip(img, 1) if lf.mirrored else img
        t0 = time.time()
        res = annotate_frame(canon, ai, AutoConfig(**{**cfg.__dict__, "frets": int(lf.guitar.get("frets", 24))}))
        dt = time.time() - t0
        pred = res.label.anchors
        cmp_ = compare(fr.anchors, pred)
        rows = cmp_["rows"]
        e_all = [e for _, e, _ in rows]
        e_v2 = [e for g, e, _ in rows if g.vis == 2]
        e_b = [e for g, e, _ in rows if g.side == "bass"]
        e_t = [e for g, e, _ in rows if g.side == "treble"]
        e_p = [p for _, _, p in rows]
        for k, v in zip(("all", "vis2", "bass", "treble", "perp"), (e_all, e_v2, e_b, e_t, e_p), strict=True):
            pooled[k] += [x for x in v if x is not None]
        cn = crossneck(fr.anchors, pred, cmp_["offset"])
        for k in cn_keys:
            pooled[k] += cn[k]
        gt_anch, gt_cue = fr.anchorability()
        pr_anch, pr_cue = res.label.anchorability()
        num = res.numbering
        r = {
            "item": f"{lf.clip_id}#{i}",
            "source": fr.source,
            "n_gt": len(rows),
            "recall": sum(e is not None for e in e_all) / max(len(rows), 1),
            "err": stats(e_all),
            "err_vis2": stats(e_v2),
            "err_bass": stats(e_b),
            "err_treble": stats(e_t),
            "err_perp": stats(e_p),
            **{f"cn_{k}": stats(cn[k]) for k in cn_keys},
            "cn_span_signed_pct_mean": float(np.mean(cn["span_signed_pct"])) if cn["span_signed_pct"] else None,
            "strings_source": res.label.extra.get("auto", {}).get("strings_source"),
            "index_offset": cmp_["offset"],
            # A relative (unresolved) numbering never counts as correct, even when the guess happens to match.
            "numbering_ok": cmp_["offset"] == 0 and bool(num and num.resolved),
            "numbering_source": num.source if num else None,
            "numbering_cue": num.cue if num else None,
            "gt_anchorable": gt_anch,
            "gt_cue": gt_cue,
            "pred_anchorable": pr_anch,
            "pred_cue": pr_cue,
            "flags": res.label.flags,
            "seconds": round(dt, 2),
        }
        results.append(r)
        if a.crops:
            a.crops.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(a.crops / f"{lf.clip_id}_f{i:03d}.jpg"),
                worst_crops(canon, rows, cmp_["offset"], pred),
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
    if not results:
        sys.exit("no GT frames")

    def f(x):
        return f"{x:.2f}"

    def num_cell(r):
        if r["numbering_ok"]:
            return "yes"
        return "unresolved" if r["numbering_source"] == "relative" else f"NO (off {r['index_offset']})"

    print(
        "\n| frame | GT anchors | recall | mean px | median | max | vis2 mean | bass mean | treble mean | "
        "perp mean | span err px | span err % | outer px | outer % | index ok | numbering | anchorable GT/pred | "
        "cue GT/pred |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['item']} | {r['n_gt']} | {r['recall']:.0%} | {f(r['err']['mean'])} | {f(r['err']['median'])} | "
            f"{f(r['err']['max'])} | {f(r['err_vis2']['mean'])} | {f(r['err_bass']['mean'])} | "
            f"{f(r['err_treble']['mean'])} | {f(r['err_perp']['mean'])} | "
            f"{f(r['cn_span_px']['mean'])} | {f(r['cn_span_pct']['mean'])} | {f(r['cn_outer_px']['mean'])} | "
            f"{f(r['cn_outer_pct']['mean'])} | {num_cell(r)} | "
            f"{r['numbering_source']}/{r['numbering_cue']} | {r['gt_anchorable']}/{r['pred_anchorable']} | "
            f"{r['gt_cue']}/{r['pred_cue']} |"
        )
    P = {k: stats(v) for k, v in pooled.items()}
    anch = [r for r in results if r["gt_anchorable"]]
    summary = {
        "frames": len(results),
        "pooled": P,
        "numbering_acc": float(np.mean([r["numbering_ok"] for r in results])),
        "numbering_acc_anchorable": float(np.mean([r["numbering_ok"] for r in anch])) if anch else None,
        "anchorable_acc": float(np.mean([r["gt_anchorable"] == r["pred_anchorable"] for r in results])),
        "cue_acc": float(np.mean([r["gt_cue"] == r["pred_cue"] for r in results])),
        "ai": ai.usage.summary() if ai else None,
    }
    print(
        f"\npooled: mean {f(P['all']['mean'])} px, median {f(P['all']['median'])}, max {f(P['all']['max'])} "
        f"(n={P['all']['n']}); vis2 mean {f(P['vis2']['mean'])}; bass {f(P['bass']['mean'])} / treble "
        f"{f(P['treble']['mean'])}; perpendicular-to-wire {f(P['perp']['mean'])}"
    )
    print(
        f"cross-neck: span width err {f(P['span_px']['mean'])} px ({f(P['span_pct']['mean'])}% of span), "
        f"outer-string err {f(P['outer_px']['mean'])} px ({f(P['outer_pct']['mean'])}%)"
    )
    print(
        f"numbering exact: {summary['numbering_acc']:.0%} (anchorable frames: "
        f"{summary['numbering_acc_anchorable'] if anch else 'n/a'}); anchorable acc {summary['anchorable_acc']:.0%}; "
        f"cue acc {summary['cue_acc']:.0%}"
    )
    if ai:
        print(f"AI usage: {summary['ai']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"summary": summary, "frames": results}, indent=1, default=str))


if __name__ == "__main__":
    main()
