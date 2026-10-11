"""Single-frame auto-annotation: frame -> `fretboard-kpts/v1` FrameLabel (same schema as samples/anchors, D12 fields).

    detect (classical CV, relative indices)
      -> [AI rescue: rough fingerboard box -> detect again inside it]   only if detection failed
      -> numbering: AI (nut / inlays / last fret) or relative fallback
      -> drop wires past the last fret / flagged as non-wires, re-measure
      -> absolute anchors (+ nut anchors when the nut is visible), refit, D12 anchorable / anchor_cue, QA flags

The optional AI usability check ("close-up electric guitar, neck clear?") runs first when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.ai import FAST_MODEL, AIClient
from gro.auto.detect import Detection, DetectParams, detect_frame, prep_gray, refit_subset
from gro.auto.numbering import Numbering, anchorability, ask_numbering, relative_numbering
from gro.geometry import FitError, FretboardModel, fit_fretboard

ANNOTATOR = "auto:gro-cv-v1"


@dataclass
class AutoConfig:
    frets: int | None = 24  # guitar.frets if known (Chase's guitars: 24)
    use_ai: bool = True
    ai_usability: bool = True
    ai_rescue: bool = True
    numbering_model: str | None = None  # default AIClient.model
    usability_model: str = FAST_MODEL
    rmse_flag_px: float = 1.5
    nut: bool = True  # add nut anchors when the AI says the nut is visible
    ai_span_check: bool = True  # AI verifier "does the grid cover all 6 strings?" (flag only, cached)
    params: DetectParams = field(default_factory=DetectParams)


@dataclass
class FrameResult:
    label: A.FrameLabel
    det: Detection | None
    numbering: Numbering | None
    model: FretboardModel | None
    usability: dict | None = None
    notes: list[str] = field(default_factory=list)


USABILITY_PROMPT = """Is this video frame usable for labelling fret positions on a guitar fingerboard? Usable means: \
an ELECTRIC guitar (not acoustic/classical), shown close up, with a substantial part of the fingerboard (at least \
~6 consecutive fret wires) clearly visible and in focus (fingers covering part of it are fine). Reply with JSON only: \
{"usable": <bool>, "guitar": "electric"|"acoustic"|"classical"|"bass"|"none"|"unclear", "close_up": <bool>, \
"neck_clear": <bool>, "visible_frets_estimate": <int>, "reason": "<one sentence>"}"""

RESCUE_PROMPT = """The image shows a guitar. Give the approximate fingerboard (the part of the neck with the fret \
wires) as a quadrilateral, in normalized image coordinates (x from the left, y from the top, both 0..1): the four \
corners in order: nut-side top, body-side top, body-side bottom, nut-side bottom. If no fingerboard is visible, \
return null corners. Reply with JSON only: {"corners": [[x, y], [x, y], [x, y], [x, y]] or null, "reason": "<short>"}"""


SPAN_PROMPT = """This is a crop of a guitar fingerboard with an overlay. The RED line is meant to lie on the low E \
string (one outermost string) and the BLUE line on the high E string (the other outermost string); thin yellow \
lines mark fret wires. Does the band between the red and blue lines cover all 6 strings along the marked frets (red \
and blue each on an outer string, within about half a string spacing)? Reply with JSON only: {"covers_all_6": \
<bool>, "problem": "none"|"misses_red_side"|"misses_blue_side"|"misses_both"|"too_wide"|"not_on_neck"|"unclear", \
"reason": "<one sentence>"}"""


def span_crop(img: np.ndarray, anchors: list[A.Anchor], pad: float = 0.25) -> np.ndarray | None:
    """Crop around the anchors with the outer-string lines (red = low E, blue = high E) and wires drawn in."""
    by: dict[int, dict] = {}
    for a in anchors:
        by.setdefault(a.fret, {})[a.side] = np.array([a.x, a.y])
    wires = [(f, d["bass"], d["treble"]) for f, d in sorted(by.items()) if len(d) == 2]
    if len(wires) < 2:
        return None
    out = img.copy()
    for _, b, t in wires:
        cv2.line(out, tuple(map(int, b)), tuple(map(int, t)), (0, 230, 255), 1, cv2.LINE_AA)
    for i, col in ((1, (0, 0, 255)), (2, (255, 80, 0))):
        cv2.line(out, tuple(map(int, wires[0][i])), tuple(map(int, wires[-1][i])), col, 2, cv2.LINE_AA)
    pts = np.array([p for w in wires for p in w[1:]])
    lo, hi = pts.min(0), pts.max(0)
    m = pad * (hi - lo).max()
    h, w = img.shape[:2]
    x0, y0 = np.maximum(lo - m, 0).astype(int)
    x1, y1 = np.minimum(hi + m, [w, h]).astype(int)
    crop = out[y0:y1, x0:x1]
    sc = min(1.0, 1024 / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else crop


def check_span(ai: AIClient, img: np.ndarray, anchors: list[A.Anchor], model: str = FAST_MODEL) -> dict | None:
    if ai is None or not ai.available:
        return None
    crop = span_crop(img, [a for a in anchors if a.fret > 0])
    return None if crop is None else ai.chat_json("span", SPAN_PROMPT, [crop], model=model, max_tokens=1500)


def check_usability(ai: AIClient, img: np.ndarray, model: str = FAST_MODEL) -> dict | None:
    if ai is None or not ai.available:
        return None
    small = cv2.resize(img, None, fx=min(1.0, 768 / max(img.shape[:2])), fy=min(1.0, 768 / max(img.shape[:2])))
    return ai.chat_json("usability", USABILITY_PROMPT, [small], model=model, max_tokens=1500)


def rescue_roi(ai: AIClient, img: np.ndarray, model: str | None = None) -> np.ndarray | None:
    """AI-provided rough fingerboard quad -> dilated ROI mask for the candidate search."""
    if ai is None or not ai.available:
        return None
    ans = ai.chat_json("rescue", RESCUE_PROMPT, [img], model=model, max_tokens=2000)
    if not ans or not ans.get("corners"):
        return None
    try:
        q = np.array(ans["corners"], float).reshape(4, 2)
    except (ValueError, TypeError):
        return None
    h, w = img.shape[:2]
    q = q * [w, h]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(q))], 255)
    k = max(15, int(0.06 * max(h, w)))
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def _nut_step(v: np.ndarray, f: np.ndarray) -> float | None:
    """Fractional fret where the profile (board side first) gets halfway from the board level to the nut block."""
    board = float(np.median(v[(f > 0.08) & (f < 0.35)]))
    far = v[(f < -0.02) & (f > -0.2)]  # the nut block itself, not the headstock beyond it
    ext = float(far[np.argmax(np.abs(far - board))])
    if abs(ext - board) <= 15:
        return None
    thr = 0.5 * (board + ext)
    side = np.sign(ext - board)
    for i in range(1, len(f)):
        if f[i] <= 0.15 and side * (v[i] - thr) >= 0:
            a, b = v[i - 1], v[i]
            w = (thr - a) / (b - a) if b != a else 0.0
            return float(f[i - 1] + w * (f[i] - f[i - 1]))
    return None


def _nut_anchors(model: FretboardModel, gray: np.ndarray) -> list[A.Anchor]:
    """Nut (fret 0) at the end of the fingerboard. Brightness is sampled along the neck for fret positions -0.4..0.4
    at 8 positions on each half of the board and median-combined per half (inlays and clamp screws only hit a few);
    on each half the nut is where the profile, scanning from the board side, gets halfway to the nut block level.
    The nut line runs through both points (it needn't be parallel to the fitted fret 0) and is cut by the board edges.
    It's a block, not a ridge, so the wire snap doesn't apply."""
    from gro.auto.detect import sample
    from gro.snap import intersect

    f = np.linspace(0.4, -0.4, 161)  # board side first
    pts = []
    for ys in (np.linspace(0.08, 0.45, 8), np.linspace(0.55, 0.92, 8)):
        rows = [sample(gray, model.project_fy(f, np.full_like(f, y))) for y in ys]
        f0 = _nut_step(np.median(np.array(rows), axis=0), f)
        pts.append(model.project_fy(0.0 if f0 is None else f0, float(ys.mean())))
    d = (pts[1] - pts[0]) / max(np.linalg.norm(pts[1] - pts[0]), 1e-9)
    out = []
    for side, y in (("bass", 0.0), ("treble", 1.0)):
        e0, e1 = model.project_fy(np.array([0.0, 1.0]), np.array([y, y]))
        p = intersect((pts[0], d), (e0, (e1 - e0) / np.linalg.norm(e1 - e0)))
        out.append(A.Anchor(0, side, float(p[0]), float(p[1]), 2))
    return out


def annotate_frame(
    img: np.ndarray,
    ai: AIClient | None,
    cfg: AutoConfig,
    t: float | None = None,
    extra: dict | None = None,
    numbering_hint=None,
) -> FrameResult:
    """`img` is in canonical orientation (mirror it first for `video.mirrored` clips, D1). `numbering_hint(det)` may
    return a Numbering (e.g. the session lock from the tracker), which is used instead of an AI call."""
    notes: list[str] = []
    flags: list[str] = []
    use_ai = cfg.use_ai and ai is not None and ai.available
    usab = None
    if use_ai and cfg.ai_usability:
        usab = check_usability(ai, img, cfg.usability_model)
        if usab is not None and not usab.get("usable", True):
            flags.append("ai_unusable")
            notes.append(f"AI usability: {usab.get('reason', '')}")
    det = detect_frame(img, cfg.params)
    if not det.ok and use_ai and cfg.ai_rescue:
        roi = rescue_roi(ai, img, cfg.numbering_model)
        if roi is not None:
            det2 = detect_frame(img, cfg.params, roi=roi)
            if det2.ok:
                det = det2
                flags.append("ai_rescued")
    base = dict(extra or {})
    if not det.ok:
        flags.append("detect_failed")
        lab = A.FrameLabel(
            t=t,
            anchors=[],
            source="pseudo",
            annotator=ANNOTATOR,
            status="rejected",
            neck_present=bool(usab and usab.get("usable")),
            flags=flags,
            notes="; ".join([*notes, f"detection failed: {det.reason}"]),
            extra=base,
        )
        lab.anchorable, lab.anchor_cue = False, None
        return FrameResult(lab, det, None, None, usab, notes)

    gray = prep_gray(img)
    numbering = numbering_hint(det) if numbering_hint is not None else None
    if numbering is None and use_ai:
        numbering = ask_numbering(ai, img, det, cfg.frets, cfg.numbering_model)
    if numbering is None:
        numbering = relative_numbering(det)
        flags.append("index_unresolved")
        if use_ai:
            notes.append(f"AI numbering failed ({ai.last_error or 'no answer'}); relative numbering")
    # Drop wires that can't be fret wires: AI-flagged ones and anything past the last fret / before the nut.
    keep = {n for n in det.wires if n not in numbering.false_wires}
    if numbering.resolved:
        keep = {n for n in keep if 1 <= n + numbering.shift <= (cfg.frets or 24)}
    elif cfg.frets:
        # Relative: at most `frets` wires; trim the weaker end (false wires sit at the body end).
        while len(keep) > cfg.frets:
            lo, hi = min(keep), max(keep)
            keep.discard(hi if det.wires[hi].score <= det.wires[lo].score else lo)
    if keep != set(det.wires) or numbering.resolved:
        # With the absolute numbering known the width prior is exact (cross-neck span vs the local fret spacing).
        det = refit_subset(gray, det, keep, cfg.params, numbering.shift if numbering.resolved else None)
    src = det.debug.get("strings_source")
    if src in ("edge_inset", "prior"):
        flags.append(f"strings_{src}")
    s = numbering.shift
    anchors = [A.Anchor(a.fret + s, a.side, a.x, a.y, a.vis) for a in det.anchors if a.fret in keep]
    try:
        model = fit_fretboard([a for a in anchors if a.vis == 2], tau=None, n_frets=cfg.frets or 24)
    except FitError as e:
        model = None
        flags.append("fit_failed")
        notes.append(str(e))
    if (
        model is not None
        and cfg.nut
        and numbering.resolved
        and numbering.nut_visible
        and min(a.fret for a in anchors) == 1
    ):
        anchors = _nut_anchors(model, gray) + anchors
    if model is not None:
        out_ids = {(a.fret, a.side) for a in model.outliers}
        if out_ids:
            flags.append("fit_outliers")
        if model.rmse_px > cfg.rmse_flag_px:
            flags.append("fit_rmse_high")
    n_vis = len({a.fret for a in anchors if a.vis == 2})
    if n_vis < 6:
        flags.append("few_wires")
    if numbering.resolved and numbering.confidence < 0.6:
        flags.append("numbering_low_conf")
    span = None
    if use_ai and cfg.ai_span_check and anchors:
        span = check_span(ai, img, anchors, cfg.usability_model)
        if span is not None and span.get("covers_all_6") is False:
            flags.append("ai_span_bad")
            notes.append(f"AI span check: {span.get('problem')}: {span.get('reason', '')}")
    anchorable, cue = anchorability(anchors, numbering)
    fit = None
    if model is not None:
        fit = {
            "H": np.round(model.H, 8).tolist(),
            "tau": model.tau,
            "rmse_px": round(model.rmse_px, 3),
            "index_shift": 0,
        }
    auto = {
        "numbering": {
            "source": numbering.source,
            "resolved": numbering.resolved,
            "cue": numbering.cue,
            "confidence": numbering.confidence,
            "nut_visible": numbering.nut_visible,
            "reasoning": numbering.reasoning,
        },
        "n_candidates": det.n_candidates,
        "grid_score": round(det.grid.score, 2) if det.grid else None,
        "strings_source": src,
        "strings_rho": [round(x, 3) for x in det.debug["strings_rho"]] if det.debug.get("strings_rho") else None,
        "usability": usab,
        "span_check": span,
    }
    lab = A.FrameLabel(
        t=t,
        anchors=sorted(anchors, key=lambda a: (a.fret, a.side)),
        source="pseudo",
        annotator=ANNOTATOR,
        status="draft",
        neck_present=True,
        fit=fit,
        flags=flags,
        notes="; ".join(notes),
        extra={**base, "auto": auto},
    )
    lab.anchorable, lab.anchor_cue = anchorable, cue
    return FrameResult(lab, det, numbering, model, usab, notes)
