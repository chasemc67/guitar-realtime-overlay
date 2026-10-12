"""Temporal auto-annotation of a clip (PLAN §2.4 / §3.2 "between keyframes").

- **Keyframes** (every `keyframe_s` seconds, and whenever the track is lost) get the full single-frame pipeline.
  The fret numbering is a per-session lock: the first keyframe asks the AI (or falls back to relative), later
  keyframes inherit it by aligning their relative grid to the propagated model (no AI call), unless they disagree.
- **Between keyframes** the model is propagated with KLT (Shi-Tomasi corners inside the board, forward-backward
  check, RANSAC homography), then every wire is re-snapped to its ridge and the model refitted (fixed tau).
- **Flags**: a frame whose fit error jumps (rmse > max(jump_k x running median, jump_px)), that loses many wires, or
  whose propagated grid disagrees with a keyframe detection gets `fit_jump` / `wires_lost` / `keyframe_mismatch`
  and `status: draft` stays for review; 3 bad frames in a row force a new keyframe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.ai import AIClient
from gro.auto.detect import Detection, anchors_from_segments, prep_gray, snap_wire_fast, wire_support
from gro.auto.numbering import Numbering
from gro.auto.pipeline import ANNOTATOR, AutoConfig, FrameResult, annotate_frame
from gro.geometry import FitError, FretboardModel, fit_fretboard, fret_u, transform_model


@dataclass
class TrackConfig:
    out_fps: float = 10.0
    keyframe_s: float = 1.0
    jump_k: float = 2.5
    jump_px: float = 1.5
    min_wire_frac: float = 0.6
    snap_search: float = 3.0
    ai_min_interval_s: float = 5.0  # re-ask the AI for numbering (or the rescue box) at most this often
    ai_max_numbering: int = 0  # max numbering asks per clip (0 = unlimited)
    ai_span_every_s: float = 0.0  # AI span check on a keyframe at most this often (0 = every keyframe with AI on)


def board_mask(model: FretboardModel, shape, f_lo: int, f_hi: int, shrink: float = 0.06) -> np.ndarray:
    u0, u1 = fret_u(max(0, f_lo - 1)), fret_u(f_hi)
    poly = model.project(np.array([u0, u1, u1, u0]), np.array([shrink, shrink, 1 - shrink, 1 - shrink]))
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 255)
    return m


def klt_homography(prev: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 6, mask=mask, blockSize=5)
    if pts is None or len(pts) < 8:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **lk)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    if good.sum() < 8:
        return None
    G, _ = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 1.5)
    return G


def resnap(
    gray: np.ndarray, model: FretboardModel, frets: list[int], cfg: TrackConfig
) -> tuple[FretboardModel | None, list[A.Anchor], int]:
    """Snap each predicted wire, anchors = snapped wire x the model's board edges, refit with the session tau."""
    h, w = gray.shape[:2]
    segs = {}
    for n in frets:
        if n == 0:
            continue
        b, t = model.project_fy(np.array([n, n]), np.array([0.0, 1.0]))
        if not all(0 <= p[0] < w and 0 <= p[1] < h for p in (b, t)):
            continue
        sn, _ = snap_wire_fast(gray, b, t, search=cfg.snap_search, step=0.5)
        med, sup = wire_support(gray, sn[0], sn[1])
        if sup >= 0.5 and med >= 4.0:
            segs[n] = (sn if (sn[0] - b) @ (sn[0] - b) < (sn[1] - b) @ (sn[1] - b) else sn[::-1], 2)
    if len(segs) < 4:
        return None, [], len(segs)
    edges = []
    for y in (0.0, 1.0):
        e0, e1 = model.project_fy(np.array([0.0, 24.0]), np.array([y, y]))
        edges.append((e0, (e1 - e0) / np.linalg.norm(e1 - e0)))
    down = model.project_fy(12, 1.0) - model.project_fy(12, 0.0)
    anchors, _ = anchors_from_segments(gray, segs, down / np.linalg.norm(down), edge_lines=tuple(edges))
    try:
        m = fit_fretboard(anchors, tau=model.tau, n_frets=model.n_frets)
    except FitError:
        return None, anchors, len(segs)
    return m, anchors, len(segs)


def _align_shift(det: Detection, model: FretboardModel, min_match: int = 5) -> tuple[int | None, float]:
    """Index shift that maps the detection's relative wires onto the propagated (absolute) model, and the fraction
    of wires that agree (segment midpoint within 25% of the local fret gap of the propagated wire line). Ambiguous
    alignments (fewer than `min_match` matches, or a runner-up shift nearly as good) return (None, 0)."""
    mids = {n: 0.5 * (w.seg[0] + w.seg[1]) for n, w in det.wires.items() if w.vis == 2}
    if not mids:
        return None, 0.0
    scores = []
    for s in range(-30, 31):
        ok, res = 0, 0.0
        for n, p in mids.items():
            f = n + s
            if not 0 <= f <= model.n_frets:
                continue
            a, b = model.fret_line(f)
            d = (b - a) / np.linalg.norm(b - a)
            # Across-wire distance (detected segments needn't be centred on the board along the wire).
            r = abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / model.fret_width_px(max(f - 1, 0))
            if r < 0.25:
                ok += 1
                res += r
        scores.append((ok, -res, s))
    scores.sort(reverse=True)
    (ok, _, s), (ok2, _, _) = scores[0], scores[1]
    if ok < min_match or ok2 > 0.7 * ok:
        return None, 0.0
    return s, ok / len(mids)


def _numbering_asks(ai: AIClient | None) -> int:
    if ai is None:
        return 0
    t = ai.usage.by_task.get("numbering") or {}
    r = ai.usage.by_task.get("rescue") or {}
    return t.get("calls", 0) + t.get("cached", 0) + r.get("calls", 0) + r.get("cached", 0) + ai.usage.failed


def annotate_clip(
    video: str,
    mirrored: bool,
    t0: float,
    t1: float,
    ai: AIClient | None,
    cfg: AutoConfig,
    tcfg: TrackConfig | None = None,
    log=print,
) -> list[A.FrameLabel]:
    T = tcfg or TrackConfig()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i0 = int(round(t0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    step = max(1, int(round(fps / T.out_fps)))
    key_every = max(1, int(round(T.keyframe_s * fps / step)))
    labels: list[A.FrameLabel] = []
    lock: Numbering | None = None
    model: FretboardModel | None = None
    frets: list[int] = []
    prev = None
    rms_hist: deque = deque(maxlen=15)
    bad_run = 0
    last_ai_t = -1e9
    last_span_t = -1e9
    n_num_asks = 0
    k = 0
    idx = i0
    while idx <= int(round(t1 * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - i0) % step:
            idx += 1
            continue
        img = cv2.flip(frame, 1) if mirrored else frame
        gray = prep_gray(img)
        t = idx / fps
        extra = {"frame_index": idx}
        is_key = model is None or k % key_every == 0 or bad_run >= 3
        key_ok = unaligned = False
        prop = None
        if model is not None and prev is not None:
            G = klt_homography(prev, gray, board_mask(model, gray.shape, min(frets), max(frets)))
            if G is not None:
                prop = transform_model(model, G)
        if is_key:

            def hint(det, prop=prop, lock=lock):
                if lock is None or not lock.resolved:
                    return None
                if prop is None:
                    log("    lock: no propagated model")
                    return None
                s, frac = _align_shift(det, prop)
                log(f"    lock align: shift {s} frac {frac:.2f}")
                if s is None or frac < 0.6:
                    return None
                return Numbering(s, True, "session_lock", lock.cue, lock.confidence, lock.nut_visible)

            # Usability is a clip-level question: ask on the first keyframe only. AI budget: numbering / rescue at
            # most every ai_min_interval_s (and ai_max_numbering per clip), the span check every ai_span_every_s.
            allow_num = t - last_ai_t >= T.ai_min_interval_s and (
                not T.ai_max_numbering or n_num_asks < T.ai_max_numbering
            )
            allow_span = t - last_span_t >= T.ai_span_every_s
            kcfg = AutoConfig(
                **{
                    **cfg.__dict__,
                    "ai_usability": cfg.ai_usability and not labels,
                    "ai_numbering": cfg.ai_numbering and allow_num,
                    "ai_rescue": cfg.ai_rescue and allow_num,
                    "ai_span_check": cfg.ai_span_check and allow_span,
                }
            )
            asks0 = _numbering_asks(ai)
            try:
                res = annotate_frame(img, ai, kcfg, t=t, extra=extra, numbering_hint=hint)
            except Exception as e:  # noqa: BLE001 - one bad frame must not stop a clip
                log(f"    keyframe error: {type(e).__name__}: {e}")
                res = None
            if _numbering_asks(ai) > asks0:
                n_num_asks += 1
                last_ai_t = t
            if res is None:
                lab = A.FrameLabel(
                    t=t, anchors=[], source="pseudo", annotator=ANNOTATOR, status="rejected", flags=["error"],
                    extra=dict(extra),
                )
                lab.anchorable, lab.anchor_cue = False, None
                res = FrameResult(lab, None, None, None)
            lab = res.label
            if (lab.extra.get("auto") or {}).get("span_check") is not None:
                last_span_t = t
            if res.numbering is not None and res.numbering.source == "ai":
                last_ai_t = t
            key_ok = res.model is not None and bool(lab.anchors)
            unaligned = key_ok and lock is not None and res.numbering is not None and not res.numbering.resolved
            key_ok = key_ok and not unaligned
            if key_ok:
                if prop is not None and lock is not None and res.numbering and res.numbering.source != "session_lock":
                    s, frac = _align_shift(res.det, prop) if res.det else (None, 0)
                    if s is not None and frac >= 0.6 and s != res.numbering.shift:
                        lab.flags.append("keyframe_mismatch")
                if res.numbering and res.numbering.source in ("ai", "manual") and lock is None:
                    lock = res.numbering
                model = res.model
                frets = sorted({a.fret for a in lab.anchors if a.vis == 2})
                rms_hist.append(model.rmse_px)
                bad_run = 0
                lab.extra["keyframe"] = True
        if not is_key or not key_ok:
            # Propagated frame (or a keyframe whose detection failed: keep tracking instead).
            try:
                m2, anchors, n_ok = (None, [], 0) if prop is None else resnap(gray, prop, frets, T)
            except Exception as e:  # noqa: BLE001
                log(f"    resnap error: {type(e).__name__}: {e}")
                m2, anchors, n_ok = None, [], 0
            flags = [] if not is_key else ["keyframe_unaligned" if unaligned else "keyframe_detect_failed"]
            if prop is None:
                flags.append("klt_lost")
            if m2 is None:
                flags.append("resnap_failed")
                lab = A.FrameLabel(
                    t=t, anchors=[], source="interp", annotator=ANNOTATOR, status="rejected", flags=flags, extra=extra
                )
                lab.anchorable, lab.anchor_cue = False, None
                bad_run += 1
                if prop is not None:
                    model = prop
            else:
                med = float(np.median(rms_hist)) if rms_hist else m2.rmse_px
                if m2.rmse_px > max(T.jump_k * med, T.jump_px):
                    flags.append("fit_jump")
                if n_ok < T.min_wire_frac * len(frets):
                    flags.append("wires_lost")
                bad_run = bad_run + 1 if flags else 0
                model = m2
                rms_hist.append(m2.rmse_px)
                fit = {
                    "H": np.round(m2.H, 8).tolist(),
                    "tau": m2.tau,
                    "rmse_px": round(m2.rmse_px, 3),
                    "index_shift": 0,
                }
                lab = A.FrameLabel(
                    t=t,
                    anchors=anchors,
                    source="interp",
                    annotator=ANNOTATOR,
                    status="draft",
                    fit=fit,
                    flags=flags,
                    extra=extra,
                )
                if lock is None or not lock.resolved:
                    lab.anchorable, lab.anchor_cue = False, None
                    lab.flags.append("index_unresolved")
        labels.append(lab)
        log(
            f"  t={t:7.2f}s {'KEY ' if is_key else 'prop'} anchors={len(lab.anchors):3d} "
            f"rmse={(lab.fit or {}).get('rmse_px', float('nan')):.2f} flags={','.join(lab.flags) or '-'}"
        )
        prev = gray
        k += 1
        idx += 1
    cap.release()
    return labels
