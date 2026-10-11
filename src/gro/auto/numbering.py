"""Absolute fret numbering for a detected (relative) fret grid, PLAN D4/D11/D12.

Fret spacing can't fix the index (an index shift is an affine map of u), so the detector's indices are relative.
The shift comes from, in order:

1. an AI call (`ask_numbering`): a rectified, zoomed neck strip with the detected wires labelled W1..Wn (nut side
   left) plus the full frame for context; the model reads the nut / inlay pattern (3-5-7-9-12-15...) / last fret
   and returns the fret number of W1;
2. otherwise the result is *relative*: indices are kept with the first detected wire = 1, the frame is flagged
   `index_unresolved` and is never anchorable (D11 graceful degradation).

`anchorability` then applies the D12 rule (nut, else 9-12-15, else all of 3-5-7 visible) to the numbered anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gro import anchors as A
from gro.auto.ai import AIClient
from gro.auto.detect import Detection
from gro.geometry import fret_u

LABEL_COLOR = (0, 255, 255)
OCC_COLOR = (0, 160, 255)


@dataclass
class Numbering:
    shift: int  # absolute fret = relative index + shift
    resolved: bool
    source: str  # "ai" | "relative" | "manual"
    cue: str | None = None
    confidence: float = 0.0
    nut_visible: bool = False
    false_wires: list[int] = field(default_factory=list)  # relative indices the AI flagged as not fret wires
    reasoning: str = ""
    raw: dict | None = None


def wire_labels(det: Detection) -> dict[int, str]:
    """Relative index -> 'W1'.. ordered from the nut side."""
    lo = min(det.wires)
    return {n: f"W{n - lo + 1}" for n in sorted(det.wires)}


def render_strip(img: np.ndarray, det: Detection, width: int = 2400, board_px: int = 240) -> np.ndarray:
    """Neck strip rectified with the fitted model (nut side left), wires labelled above and ticked outside the board
    so inlays stay visible."""
    m = det.model
    lo, hi = min(det.wires), max(det.wires)
    u0, u1 = float(fret_u(lo - 3.2)), float(fret_u(hi + 1.8))
    y0, y1 = -0.75, 1.75
    height = int(board_px * (y1 - y0))
    xs = np.linspace(u0, u1, width)
    ys = np.linspace(y0, y1, height)
    U, Y = np.meshgrid(xs, ys)
    P = m.project(U, Y).astype(np.float32)
    out = cv2.remap(
        img, P[..., 0], P[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40)
    )
    labels = wire_labels(det)

    def col(n):
        return int(round((fret_u(n) - u0) / (u1 - u0) * (width - 1)))

    def row(y):
        return int(round((y - y0) / (y1 - y0) * (height - 1)))

    for n, w in det.wires.items():
        x = col(n)
        c = LABEL_COLOR if w.vis == 2 else OCC_COLOR
        for ya, yb in ((-0.22, -0.04), (1.04, 1.22)):
            if w.vis == 2:
                cv2.line(out, (x, row(ya)), (x, row(yb)), c, 2, cv2.LINE_AA)
            else:
                for k in range(3):
                    t0 = ya + (yb - ya) * k / 3
                    cv2.line(out, (x, row(t0)), (x, row(t0 + (yb - ya) / 6)), c, 2, cv2.LINE_AA)
        s = labels[n]
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        yy = row(-0.3) if (n - lo) % 2 == 0 else row(-0.45)
        cv2.putText(out, s, (x - tw // 2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, s, (x - tw // 2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return out


def render_context(img: np.ndarray, det: Detection, max_side: int = 1400) -> np.ndarray:
    out = img.copy()
    labels = wire_labels(det)
    for n, w in det.wires.items():
        b, t = w.seg
        c = LABEL_COLOR if w.vis == 2 else OCC_COLOR
        cv2.line(out, tuple(np.int32(np.round(b))), tuple(np.int32(np.round(t))), c, 1, cv2.LINE_AA)
        k = n - min(det.wires)
        if k % 3 == 0 or n == max(det.wires):
            d = (b - t) / max(np.linalg.norm(b - t), 1e-6)
            p = b + d * (14 + 10 * ((k // 3) % 2))
            cv2.putText(out, labels[n], tuple(np.int32(p - [12, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, labels[n], tuple(np.int32(p - [12, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    h, w_ = out.shape[:2]
    k = min(1.0, max_side / max(h, w_))
    return cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else out


NUMBERING_SYSTEM = (
    "You are a meticulous guitar-luthier assistant helping label fret positions in video frames for a computer "
    "vision dataset. Answer only from what is visible. Reply with a single JSON object."
)

NUMBERING_PROMPT = (
    "Image 1 is a rectified, zoomed strip of an electric guitar fingerboard: the NUT / HEADSTOCK side is "
    "on the LEFT and the BODY side on the RIGHT. Every detected fret wire is marked by short ticks above "
    "and below the fingerboard and labelled W1, W2, ... from left to right (dashed orange ticks = wire "
    "hidden under a finger, position inferred). Image 2 is the whole frame for context, with the same "
    "wires drawn and some labelled (it may be horizontally mirrored, so ignore text direction).\n\nTask: "
    "give the absolute fret number of wire W1. The nut is fret 0; the first fret wire after the nut is "
    "fret 1.\n\nCues, strongest first:\n(a) The nut: the bar (bone/plastic/graphite, or a locking-nut clamp "
    "block) where the fingerboard meets the headstock. A fret wrap / string dampener (a band around the "
    "neck, often black) may sit between the nut and fret 1 and hide the nut; frets still count from the "
    "nut under it.\n(b) Position markers (inlays): standard markers sit at frets 3, 5, 7, 9, 12 (double or"
    " larger), 15, 17, 19, 21, 24. The marker of fret n lies in the cell BETWEEN wires n-1 and n. With "
    "ornate/vine inlays, look for the distinctive 12th-fret motif and the regular odd-fret spacing of the"
    " larger shapes. Side dots on the edge of the neck follow the same pattern.\n(c) The last fret: "
    "{frets_line} The fingerboard end and pickup edges can be mis-detected as extra wires near the body, "
    "so do not just count the last label.\nCount carefully wire by wire from your anchor cue to W1.\n\n"
    'Return JSON only:\n{{"reasoning": "<2-5 sentences: which cue, which labelled wire it identifies, and '
    'the counting>", "w1_fret": <int or null if it cannot be determined>, "nut_visible": <bool: the nut '
    'bar itself is visible, not covered by a fret wrap or hand and not outside the image>, "fret_wrap": '
    '<bool: a fret wrap / dampener band is on the neck>, "cue": "nut" | "inlay_9_12_15" | "inlay_3_5_7" |'
    ' "last_fret" | "other" | null, "confidence": <0..1>, "false_wires": [<labels that are not real fret '
    "wires, e.g. the fingerboard end>]}}"
)


def ask_numbering(
    ai: AIClient, img: np.ndarray, det: Detection, frets: int | None = None, model: str | None = None
) -> Numbering | None:
    if ai is None or not ai.available or not det.ok or not det.wires:
        return None
    strip = render_strip(img, det)
    ctx = render_context(img, det)
    frets_line = (
        f"this guitar has {frets} frets, so the last real wire before the fingerboard ends at the body is fret {frets}."
        if frets
        else "the fret count is unknown (usually 21, 22 or 24)."
    )
    ans = ai.chat_json(
        "numbering",
        NUMBERING_PROMPT.format(frets_line=frets_line),
        [strip, ctx],
        model=model,
        system=NUMBERING_SYSTEM,
    )
    if not ans or ans.get("w1_fret") is None:
        return None
    try:
        w1 = int(ans["w1_fret"])
    except (TypeError, ValueError):
        return None
    lo = min(det.wires)
    labels = {v: k for k, v in wire_labels(det).items()}
    false = [labels[s] for s in ans.get("false_wires") or [] if s in labels]
    return Numbering(
        shift=w1 - lo,
        resolved=True,
        source="ai",
        cue=ans.get("cue"),
        confidence=float(ans.get("confidence") or 0.0),
        # A wrap sits right after the nut; models tend to call the clamp block "the nut" anyway.
        nut_visible=bool(ans.get("nut_visible")) and not bool(ans.get("fret_wrap")),
        false_wires=false,
        reasoning=str(ans.get("reasoning", "")),
        raw=ans,
    )


def relative_numbering(det: Detection) -> Numbering:
    """No absolute cue: the first detected wire is called fret 1 (D11 fallback, flagged and never anchorable)."""
    return Numbering(shift=1 - min(det.wires), resolved=False, source="relative")


def anchorability(anchors: list[A.Anchor], numbering: Numbering) -> tuple[bool, str | None]:
    """D12 on the numbered anchors; a relative (unresolved) numbering is never anchorable."""
    if not numbering.resolved:
        return False, None
    fl = A.FrameLabel(t=None, anchors=anchors)
    return fl.anchorability()
