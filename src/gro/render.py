"""Overlay renderer ("reference" style, PLAN §7) with OpenCV.

Everything is defined in canonical fretboard coordinates and projected through the fitted model, so perspective is
free. Strings are addressed by physical identity (0 = low E), so the left-handed / mirrored-camera toggle only changes
the homography (H_img = M . H_canon, see `render_on_original`), never the drawing code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from gro.geometry import NUT_WIDTH_U, FretboardModel, fret_u, mirror_model, string_y
from gro.theory import ScaleNote

SINGLE_INLAYS = (3, 5, 7, 9, 15, 17, 19, 21)
DOUBLE_INLAYS = (12, 24)

# BGR
BLUE = (225, 120, 30)
STRING_GRAY = (225, 225, 225)
WHITE = (255, 255, 255)
YELLOW = (0, 225, 255)
GREEN = (60, 220, 60)
BLACK = (0, 0, 0)


@dataclass
class Style:
    cell_color: tuple[int, int, int] = BLUE
    cell_alpha: float = 0.38
    cell_gap: float = 0.06  # fraction of cell length left open at each wire so the wires show through
    string_color: tuple[int, int, int] = STRING_GRAY
    string_alpha: float = 0.6
    inlay_color: tuple[int, int, int] = WHITE
    inlay_radius: float = 0.11  # in nut widths
    note_color: tuple[int, int, int] = YELLOW
    root_color: tuple[int, int, int] = GREEN
    note_radius: float = 0.1  # in nut widths
    note_marker_pos: float = 0.6  # along the cell, 0 = lower wire, 1 = upper wire (where a finger presses)
    label: str = "name"  # "name" | "degree" | "none"
    min_label_px: float = 11.0  # skip text on markers smaller than this
    outline_cells: bool = True


REFERENCE = Style()


def _poly(img, pts, color, thickness=-1):
    p = np.round(np.asarray(pts) * 16).astype(np.int32).reshape(-1, 1, 2)
    if thickness < 0:
        cv2.fillPoly(img, [p], color, lineType=cv2.LINE_AA, shift=4)
    else:
        cv2.polylines(img, [p], True, color, thickness, lineType=cv2.LINE_AA, shift=4)


def _line(img, a, b, color, thickness=1):
    a = tuple(np.round(np.asarray(a) * 16).astype(int))
    b = tuple(np.round(np.asarray(b) * 16).astype(int))
    cv2.line(img, a, b, color, thickness, lineType=cv2.LINE_AA, shift=4)


def _disc(model: FretboardModel, u: float, y: float, r_nut: float, n: int = 24) -> np.ndarray:
    """Image polygon of a circle of radius r_nut (nut widths) centered at canonical (u, y), in perspective."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 1.0 + model.tau * u  # local width in nut widths
    uu = u + r_nut * NUT_WIDTH_U * np.cos(th)
    yy = y + r_nut * np.sin(th) / width
    return model.project(uu, yy)


def _blend(dst, layer, mask, alpha):
    m = mask > 0
    dst[m] = (dst[m] * (1 - alpha) + layer[m] * alpha).astype(np.uint8)


def default_fret_range(model: FretboardModel, shape) -> tuple[int, int]:
    """Frets with any part in view, padded per D4 ([min_visible-1, max_visible+1]), clamped to [1, n_frets]."""
    h, w = shape[:2]
    vis = model.visible_frets(w, h, pad=0.0)
    if not vis:
        return 1, 0
    return max(1, min(vis)), min(model.n_frets, max(vis) + 1)


def render_overlay(
    frame: np.ndarray,
    model: FretboardModel,
    notes: Sequence[ScaleNote] = (),
    style: Style = REFERENCE,
    fret_range: tuple[int, int] | None = None,
    show_open: bool = True,
) -> np.ndarray:
    """Draw the overlay onto a copy of `frame` (BGR). `fret_range` = (first, last) fret *cell* to draw."""
    out = frame.copy()
    f0, f1 = fret_range if fret_range is not None else default_fret_range(model, frame.shape)
    if f1 < f0:
        return out
    h, w = frame.shape[:2]

    # 1. Cells: one translucent quad per fret, full width across the strings.
    layer = np.zeros_like(out)
    mask = np.zeros((h, w), np.uint8)
    for n in range(f0, f1 + 1):
        q = model.cell_quad(n, 0.0, 1.0, inset_u=style.cell_gap)
        _poly(layer, q, style.cell_color)
        _poly(mask, q, 255)
    _blend(out, layer, mask, style.cell_alpha)

    # 2. Strings: thin light lines over the drawn range (from the nut if it's in range).
    layer = out.copy()
    start = 0.0 if f0 <= 1 else f0 - 1
    for s in range(model.n_strings):
        a, b = model.string_line(s, start, f1)
        _line(layer, a, b, style.string_color, 1)
    cv2.addWeighted(layer, style.string_alpha, out, 1 - style.string_alpha, 0, dst=out)

    # 3. Inlay dots (standard positions; a style choice even for vine inlays, PLAN §7).
    for n in range(f0, f1 + 1):
        uc = 0.5 * (fret_u(n - 1) + fret_u(n))
        if n in SINGLE_INLAYS:
            _poly(out, _disc(model, uc, 0.5, style.inlay_radius), style.inlay_color)
        elif n in DOUBLE_INLAYS:
            for y in (0.3, 0.7):
                _poly(out, _disc(model, uc, y, style.inlay_radius), style.inlay_color)

    # 4. Scale notes.
    for note in notes:
        if note.fret == 0:
            if not show_open or f0 > 1:
                continue
            u = -0.35 * (fret_u(1) - fret_u(0))  # just behind the nut
            ys = float(string_y(note.string, model.n_strings))
            _draw_note(out, model, note, u, ys, style)
            continue
        if not f0 <= note.fret <= f1:
            continue
        ys = float(string_y(note.string, model.n_strings))
        half = 0.5 * float(string_y(1, model.n_strings) - string_y(0, model.n_strings))
        color = style.root_color if note.is_root else style.note_color
        if style.outline_cells:
            y0 = 0.0 if note.string == 0 else ys - half
            y1 = 1.0 if note.string == model.n_strings - 1 else ys + half
            thick = 2 if model.fret_width_px(note.fret - 1, ys) >= 22 else 1
            _poly(out, model.cell_quad(note.fret, y0, y1, inset_u=style.cell_gap), color, thick)
        ua, ub = fret_u(note.fret - 1), fret_u(note.fret)
        u = ua + style.note_marker_pos * (ub - ua)
        _draw_note(out, model, note, u, ys, style)
    return out


def _draw_note(img, model, note: ScaleNote, u, ys, style: Style):
    color = style.root_color if note.is_root else style.note_color
    disc = _disc(model, u, ys, style.note_radius)
    _poly(img, disc, color)
    _poly(img, disc, BLACK, 1)
    if style.label == "none":
        return
    text = note.name if style.label == "name" else note.degree_name
    c = model.project(u, ys)
    diam = float(min(np.ptp(disc[:, 1]), np.ptp(disc[:, 0])))
    if diam < style.min_label_px:
        return  # too small to read; the colour still marks root vs. scale note
    scale = min(0.6, diam / 24.0) * (0.8 if len(text) > 1 else 1.0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    org = (int(round(c[0] - tw / 2)), int(round(c[1] + th / 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------------------------------
# Left-handed / mirrored-camera toggle (D1)


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


def render_on_original(frame: np.ndarray, canon_model: FretboardModel, mirrored: bool, **kw) -> np.ndarray:
    """Render using a model fitted in canonical space onto the *original* frame.

    Composes the mirror into the homography (H_img = M . H_canon) instead of drawing on the mirrored frame, so note
    labels stay readable.
    """
    model = mirror_model(canon_model, frame.shape[1]) if mirrored else canon_model
    return render_overlay(frame, model, **kw)


def with_mirror_toggle(frame: np.ndarray, mirrored: bool, process: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """App-level helper: mirror the frame into canonical orientation, run `process`, then un-mirror its image output.

    Use for pipelines that produce an image in canonical space (e.g. debug views). For the overlay itself prefer
    `render_on_original`, which avoids mirrored text.
    """
    if not mirrored:
        return process(frame)
    return mirror_frame(process(mirror_frame(frame)))


def draw_anchors(img: np.ndarray, anchors, color=(0, 0, 255), label: bool = True) -> np.ndarray:
    """Debug: draw anchor points (in the image's own coordinates)."""
    out = img.copy()
    for a in anchors:
        p = (int(round(a.x)), int(round(a.y)))
        cv2.circle(out, p, 3, color, 1, cv2.LINE_AA)
        if label and a.side == "bass":
            cv2.putText(out, str(a.fret), (p[0] - 4, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out
