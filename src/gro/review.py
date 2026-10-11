"""Annotation review images (PLAN D8): every annotation batch gets contact sheets that Chase reviews.

For each annotated frame in a batch of `fretboard-kpts/v1` label files this renders, in the frame's *original*
orientation (mirrored clips are un-mirrored with `H_img = M . H_canon`, D1):

- the clicked anchors, labelled `<fret><b|t>` and coloured by visibility (green = 2, orange = 1, red X = rejected
  by the robust fit);
- the refitted grid: every fret wire 0..`guitar.frets` and the 6 string lines, solid inside the annotated fret range
  and dashed where extrapolated, plus the two board edges;
- a caption bar: file, frame, inlier mean/max px error, rejected points, nut visible, mirrored, tau.

Outputs under `review/<batch>/`: `frames/<item>.jpg` (full res), `strips/<item>.jpg` (zoomed, neck-aligned crops,
raw above overlay, so wire alignment can be judged), `contact_sheet_NN.jpg` (tiled crops of the board) and
`summary.csv`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.geometry import FitError, FretboardModel, fit_fretboard, mirror_model, string_y, transform_model

# BGR
VIS_COLORS = {A.VIS_VISIBLE: (60, 220, 60), A.VIS_OCCLUDED: (0, 165, 255)}
REJECT_COLOR = (40, 40, 255)
FRET_COLOR = (255, 230, 0)
NUT_COLOR = (255, 0, 255)
STRING_COLOR = (230, 230, 230)
EDGE_COLOR = (0, 200, 255)
CAPTION_BG = (32, 32, 32)
WARN_BG = (0, 90, 150)
FAIL_BG = (20, 20, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX

LEGEND = (
    "anchors: green = visible, orange = occluded (vis 1), red X = rejected | grid: cyan = fret wires (magenta = nut), "
    "white = strings, yellow = board edges; solid = within annotated frets, dashed = extrapolated"
)


@dataclass
class ReviewItem:
    """One annotated frame plus its refit and stats."""

    label_path: Path
    frame_idx: int
    lf: A.LabelFile
    frame: A.FrameLabel
    image: np.ndarray | None = None  # original orientation (BGR)
    image_src: str = ""
    model: FretboardModel | None = None  # canonical space
    error: str = ""
    residuals: dict[tuple[int, str], float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.lf.clip_id}_f{self.frame_idx:03d}"

    @property
    def frame_desc(self) -> str:
        parts = [f"frame #{self.frame_idx}"]
        if self.frame.t is not None:
            parts.append(f"t={self.frame.t:g}s")
        if "frame_index" in self.frame.extra:
            parts.append(f"video frame {self.frame.extra['frame_index']}")
        return " ".join(parts)

    def stats(self) -> dict:
        m = self.model
        inl = {(a.fret, a.side) for a in m.inliers} if m else set()
        r_in = {k: r for k, r in self.residuals.items() if k in inl}
        worst = max(r_in, key=r_in.get) if r_in else None
        rejected = [] if m is None else sorted(m.outliers, key=lambda a: (a.fret, a.side))
        nut = [a for a in self.frame.present() if a.fret == 0]
        return {
            "item": self.name,
            "file": self.label_path.name,
            "frame": self.frame_desc,
            "n_anchors": len(self.frame.present()),
            "frets": _fret_span(self.frame),
            "mean_px": float(np.mean(list(r_in.values()))) if r_in else float("nan"),
            "max_px": r_in[worst] if r_in else float("nan"),
            "max_at": f"{worst[0]}{worst[1][0]}" if worst else "",
            "rmse_px": m.rmse_px if m else float("nan"),
            "n_rejected": len(rejected),
            "rejected": " ".join(f"{a.fret}{a.side[0]}" for a in rejected),
            "nut_visible": "yes" if any(a.vis == 2 for a in nut) else ("occluded" if nut else "no"),
            "mirrored": self.lf.mirrored,
            "tau": m.tau if m else float("nan"),
            "error": self.error,
            "status": self.frame.status,
        }


def _fret_span(fr: A.FrameLabel) -> str:
    frets = sorted({a.fret for a in fr.present()})
    return f"{frets[0]}-{frets[-1]}" if frets else "-"


# ---------------------------------------------------------------------------------------------------------------
# Loading


def _resolve(p: str | Path, bases: list[Path]) -> Path | None:
    p = Path(p)
    if p.is_absolute():
        return p if p.exists() else None
    for b in bases:
        if (b / p).exists():
            return b / p
    return None


def _read_video_frame(path: Path, frame: A.FrameLabel, fps: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        if "frame_index" in frame.extra:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame.extra["frame_index"]))
        elif frame.t is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(frame.t) * 1000.0)
        else:
            return None
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def load_image(
    lf: A.LabelFile, fr: A.FrameLabel, frame_idx: int, bases: list[Path], frames_dir: Path | None
) -> tuple[np.ndarray | None, str]:
    """Find the frame image (original orientation): `frame_file` in the label, then `<frames_dir>/<clip_id>[_fNNN]`
    .jpg/.png, then the clip video at `frame_index` (or `t`)."""
    cands: list[Path] = []
    if fr.extra.get("frame_file"):
        p = _resolve(fr.extra["frame_file"], bases)
        if p:
            cands.append(p)
    if frames_dir is not None:
        for stem in (f"{lf.clip_id}_f{frame_idx:03d}", lf.clip_id):
            cands += [frames_dir / f"{stem}{ext}" for ext in (".jpg", ".png")]
    for p in cands:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img, str(p)
    vp = _resolve(lf.video.get("path", ""), bases) if lf.video.get("path") else None
    if vp is not None:
        img = _read_video_frame(vp, fr, float(lf.video.get("fps", 60)))
        if img is not None:
            return img, f"{vp}@{fr.extra.get('frame_index', fr.t)}"
    return None, ""


def fit_frame(fr: A.FrameLabel, n_frets: int) -> FretboardModel:
    anchors = fr.present()
    # Taper is only observable with many frets spread along the neck (same rule as scripts/render_demo.py).
    tau = None if len({a.fret for a in anchors}) >= 8 else 0.3
    return fit_fretboard(anchors, tau=tau, n_frets=n_frets)


def load_batch(label_dir: Path, bases: list[Path], frames_dir: Path | None = None) -> list[ReviewItem]:
    items = []
    for path in sorted(Path(label_dir).glob("*.json")):
        lf = A.load(path)
        for i, fr in enumerate(lf.frames):
            if not fr.neck_present or not fr.present():
                continue
            it = ReviewItem(path, i, lf, fr)
            it.image, it.image_src = load_image(lf, fr, i, [path.parent, *bases], frames_dir)
            if it.image is None:
                it.error = "frame image not found"
            try:
                it.model = fit_frame(fr, int(lf.guitar.get("frets", 24)))
                res = it.model.residuals(fr.present())
                it.residuals = {(a.fret, a.side): float(r) for a, r in zip(fr.present(), res, strict=True)}
            except FitError as e:
                it.error = f"fit failed: {e}"
            items.append(it)
    return items


# ---------------------------------------------------------------------------------------------------------------
# Drawing primitives


def _clip(img, a, b) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = img.shape[:2]
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    big = 4 * max(w, h)  # keep cv2.clipLine's int conversion safe
    a, b = np.clip(a, -big, big), np.clip(b, -big, big)
    ok, p, q = cv2.clipLine((0, 0, w, h), tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b))
    return (p, q) if ok else None


def draw_segment(img, a, b, color, thickness=1, dashed=False, dash=7.0, gap=5.0):
    seg = _clip(img, a, b)
    if seg is None:
        return
    p, q = np.array(seg[0], float), np.array(seg[1], float)
    if not dashed:
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), color, thickness, cv2.LINE_AA)
        return
    L = float(np.linalg.norm(q - p))
    if L < 1:
        return
    d = (q - p) / L
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        cv2.line(
            img,
            tuple(map(int, np.round(p + d * s))),
            tuple(map(int, np.round(p + d * e))),
            color,
            thickness,
            cv2.LINE_AA,
        )
        s = e + gap


def _text(img, s, org, scale, color, thickness=1, outline=True):
    org = (int(round(org[0])), int(round(org[1])))
    if outline:
        cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _centered_text(img, s, center, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thickness)
    _text(img, s, (center[0] - tw / 2, center[1] + th / 2), scale, color, thickness)


def draw_grid(
    img, model: FretboardModel, lo: int, hi: int, thickness: int = 1, label_scale: float = 0.0, skip_labels=()
):
    """Every fret wire 0..n_frets (+ edges, strings) of an image-space model. Solid in [lo, hi], dashed outside.
    With `label_scale` > 0, fret numbers (except `skip_labels`) are written just beyond the treble edge."""
    n = model.n_frets
    for s in range(model.n_strings):
        y = float(string_y(s, model.n_strings))
        for f0, f1, dashed in ((0, lo, True), (lo, hi, False), (hi, n, True)):
            if f1 > f0:
                a, b = model.project_fy(np.array([f0, f1]), np.array([y, y]))
                draw_segment(img, a, b, STRING_COLOR, 1, dashed)
    for y in (0.0, 1.0):
        for f0, f1, dashed in ((0, lo, True), (lo, hi, False), (hi, n, True)):
            if f1 > f0:
                a, b = model.project_fy(np.array([f0, f1]), np.array([y, y]))
                draw_segment(img, a, b, EDGE_COLOR, 1, dashed)
    for f in range(n + 1):
        a, b = model.fret_line(f)
        draw_segment(img, a, b, NUT_COLOR if f == 0 else FRET_COLOR, thickness, not lo <= f <= hi)
        if label_scale > 0 and f not in skip_labels:
            p = model.project_fy(f, 1.0 + 0.35)
            h, w = img.shape[:2]
            if 0 <= p[0] < w and 0 <= p[1] < h:
                _centered_text(img, str(f), p, label_scale, NUT_COLOR if f == 0 else FRET_COLOR)


def draw_anchor_marks(img, anchors, model: FretboardModel | None, rejected: set, radius=4, label_scale=0.4):
    """Anchors in image coords, labelled `<fret><b|t>` on the outside of the board, coloured by visibility."""
    for a in anchors:
        p = np.array([a.x, a.y])
        ip = tuple(int(round(v)) for v in p)
        bad = (a.fret, a.side) in rejected
        color = REJECT_COLOR if bad else VIS_COLORS.get(a.vis, (200, 200, 200))
        if bad:
            r = radius + 2
            cv2.line(img, (ip[0] - r, ip[1] - r), (ip[0] + r, ip[1] + r), color, 2, cv2.LINE_AA)
            cv2.line(img, (ip[0] - r, ip[1] + r), (ip[0] + r, ip[1] - r), color, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, ip, radius, color, 1, cv2.LINE_AA)
            cv2.circle(img, ip, 1, color, -1)
        if label_scale <= 0:
            continue
        if model is not None:
            c = model.project_fy(a.fret, 0.5)
            d = p - c
            d = d / max(np.linalg.norm(d), 1e-6)
        else:
            d = np.array([0.0, -1.0 if a.side == "bass" else 1.0])
        lab = f"{a.fret}{a.side[0]}"
        (tw, th), _ = cv2.getTextSize(lab, FONT, label_scale, 1)
        off = radius + 4 + 0.5 * max(tw, th)
        _centered_text(img, lab, p + d * off, label_scale, color)


def caption_bar(width: int, lines: list[str], scale: float, bg=CAPTION_BG) -> np.ndarray:
    th = cv2.getTextSize("Ag", FONT, scale, 1)[0][1]
    lh = int(th * 1.9)
    bar = np.full((lh * len(lines) + int(th * 0.8), width, 3), bg, np.uint8)
    for i, s in enumerate(lines):
        # Shrink lines that would overflow.
        sc = scale
        while sc > 0.25 and cv2.getTextSize(s, FONT, sc, 1)[0][0] > width - 12:
            sc *= 0.92
        cv2.putText(bar, s, (6, lh * (i + 1)), FONT, sc, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def caption_lines(it: ReviewItem) -> list[str]:
    s = it.stats()
    l1 = f"{s['file']} | {s['frame']} | {s['n_anchors']} anchors, frets {s['frets']} | status {s['status']}"
    if it.model is None:
        return [l1, f"FIT: {it.error or 'n/a'}"]
    l2 = (
        f"err mean {s['mean_px']:.2f}px max {s['max_px']:.2f}px @{s['max_at']} (rmse {s['rmse_px']:.2f}) | "
        f"rejected {s['n_rejected']}{' (' + s['rejected'] + ')' if s['rejected'] else ''} | "
        f"nut visible: {s['nut_visible']} | mirrored: {'yes' if s['mirrored'] else 'no'} | tau {s['tau']:.2f}"
    )
    return [l1, l2] + ([it.error] if it.error else [])


def caption_bg(it: ReviewItem, warn_px: float) -> tuple[int, int, int]:
    if it.model is None or it.image is None:
        return FAIL_BG
    s = it.stats()
    return WARN_BG if (s["n_rejected"] or s["max_px"] > warn_px) else CAPTION_BG


# ---------------------------------------------------------------------------------------------------------------
# Per-item renders


def image_space(it: ReviewItem) -> tuple[FretboardModel | None, list[A.Anchor], set]:
    """Model, anchors and rejected keys in the original image orientation."""
    w = it.image.shape[1]
    anchors = it.frame.present()
    model = it.model
    if it.lf.mirrored:
        anchors = A.mirror_anchors(anchors, w)
        model = mirror_model(model, w) if model is not None else None
    rejected = {(a.fret, a.side) for a in it.model.outliers} if it.model else set()
    return model, anchors, rejected


def annotated_range(it: ReviewItem) -> tuple[int, int]:
    frets = [a.fret for a in it.frame.present()]
    return min(frets), max(frets)


def render_annotated(it: ReviewItem, thickness: int = 1, label_scale: float = 0.4) -> np.ndarray:
    """Full-res frame with grid + anchors (no caption)."""
    out = it.image.copy()
    model, anchors, rejected = image_space(it)
    if model is not None:
        labelled = {a.fret for a in anchors if a.side == "treble"}
        draw_grid(out, model, *annotated_range(it), thickness, label_scale, skip_labels=labelled)
    draw_anchor_marks(out, anchors, model, rejected, label_scale=label_scale)
    return out


def render_full(it: ReviewItem, warn_px: float = 3.0) -> np.ndarray:
    if it.image is None:
        img = np.full((360, 640, 3), 60, np.uint8)
    else:
        img = render_annotated(it)
    scale = max(0.45, img.shape[1] / 2400)
    return np.vstack([caption_bar(img.shape[1], caption_lines(it), scale, caption_bg(it, warn_px)), img])


def board_bbox(it: ReviewItem, pad: float = 0.12) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the anchors and the whole fitted board (clipped to the frame), padded."""
    h, w = it.image.shape[:2]
    model, anchors, _ = image_space(it)
    pts = [[a.x, a.y] for a in anchors]
    if model is not None:
        f = np.arange(model.n_frets + 1)
        for y in (-0.2, 1.2):
            pts += model.project_fy(f, np.full(len(f), y)).tolist()
    pts = np.array(pts, float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = np.clip(pts, [0, 0], [w - 1, h - 1])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    px, py = pad * (x1 - x0) + 10, pad * (y1 - y0) + 10
    return (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py)))


def neck_strip(
    it: ReviewItem, zoom: float = 3.0, max_width: int = 2000, frets_per_row: int | None = None
) -> np.ndarray | None:
    """Zoomed, neck-aligned crops: the neck axis is rotated horizontal and scaled by `zoom`, then cut into rows of at
    most `max_width` px. Each row shows the raw pixels above the same crop with the grid drawn (thin lines), so the
    fitted wires can be compared with the real ones."""
    if it.image is None or it.model is None:
        return None
    model, anchors, rejected = image_space(it)
    h, w = it.image.shape[:2]
    lo, hi = annotated_range(it)
    vis = model.visible_frets(w, h)
    f_a = max(0, min([lo - 1, *vis]))
    f_b = min(model.n_frets, max([hi + 1, *vis]))
    p0, p1 = model.project_fy(np.array([f_a, f_b]), np.array([0.5, 0.5]))
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 10:
        return None
    ang = math.atan2(d[1], d[0])
    widths = [np.linalg.norm(np.subtract(*model.fret_line(f))) for f in (f_a, f_b)]
    half_h = 0.5 * max(widths) * 1.5 * zoom
    margin = 0.5 * model.fret_width_px(max(f_a - 1, 0)) * zoom + 20
    W = int(L * zoom + 2 * margin)
    Hh = int(2 * half_h)
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s], [s, c]]) * zoom
    t = np.array([margin, half_h]) - R @ p0
    G = np.vstack([np.c_[R, t], [0, 0, 1]])
    raw = cv2.warpAffine(it.image, G[:2], (W, Hh), flags=cv2.INTER_CUBIC, borderValue=(40, 40, 40))
    zm = transform_model(model, G)
    over = raw.copy()
    draw_grid(over, zm, lo, hi, thickness=1, label_scale=0.0)
    za = [A.Anchor(a.fret, a.side, *(G[:2] @ [a.x, a.y, 1.0]), a.vis, a.conf) for a in anchors]
    draw_anchor_marks(over, za, zm, rejected, radius=6, label_scale=0.0)
    over = cv2.addWeighted(over, 0.75, raw, 0.25, 0)
    # Fret numbers along the top of the raw crop.
    for f in range(f_a, f_b + 1):
        x = zm.project_fy(f, 0.5)[0]
        if 0 <= x < W:
            _centered_text(raw, str(f), (x, 14), 0.5, NUT_COLOR if f == 0 else FRET_COLOR)
    pair = np.vstack([raw, np.full((3, W, 3), 255, np.uint8), over])
    if frets_per_row:
        cuts = [zm.project_fy(f, 0.5)[0] for f in range(f_a, f_b + 1, frets_per_row)]
    else:
        n = math.ceil(W / max_width)
        cuts = [W * i / n for i in range(n)]
    rows = []
    bounds = [*map(int, cuts), W]
    for x0, x1 in zip(bounds[:-1], bounds[1:], strict=True):
        x0, x1 = max(0, x0 - 24), min(W, x1 + 24)  # small overlap so no wire is cut at a seam
        if x1 - x0 < 30:
            continue
        r = pair[:, x0:x1]
        if r.shape[1] > max_width:
            r = cv2.resize(r, (max_width, int(r.shape[0] * max_width / r.shape[1])), interpolation=cv2.INTER_AREA)
        rows.append(np.pad(r, ((0, 10), (0, max_width - r.shape[1]), (0, 0)), constant_values=24))
    strip = np.vstack(rows)
    cap = caption_bar(strip.shape[1], [f"{it.name}: zoom x{zoom:g}, frets {f_a}-{f_b}; raw above, fit below"], 0.6)
    return np.vstack([cap, strip])


def render_tile(it: ReviewItem, tile_w: int, tile_h: int, warn_px: float = 3.0) -> np.ndarray:
    """Contact-sheet tile: crop around the board, scaled to fit, with a compact caption."""
    cap = caption_bar(tile_w, caption_lines(it), 0.42, caption_bg(it, warn_px))
    body_h = max(1, tile_h - cap.shape[0])
    body = np.full((body_h, tile_w, 3), 48, np.uint8)
    if it.image is not None:
        x0, y0, x1, y1 = board_bbox(it)
        # Draw at a line weight that survives the downscale.
        k = min(tile_w / (x1 - x0), body_h / (y1 - y0))
        th = 1 if k > 0.6 else 2
        full = render_annotated(it, thickness=th, label_scale=0.4 / min(k, 1.0) ** 0.5 * 0.8)
        crop = full[y0:y1, x0:x1]
        nw, nh = max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        oy, ox = (body_h - nh) // 2, (tile_w - nw) // 2
        body[oy : oy + nh, ox : ox + nw] = crop
    return np.vstack([cap, body])


def contact_sheets(
    items: list[ReviewItem], batch: str, cols: int = 3, rows: int = 3, width: int = 2000, warn_px: float = 3.0
) -> list[np.ndarray]:
    gap = 6
    tile_w = (width - gap * (cols + 1)) // cols
    # Tile aspect follows the (median) board crop, so the board fills the tile; +~50 px for the caption.
    aspects = [(b[3] - b[1]) / max(b[2] - b[0], 1) for b in (board_bbox(it) for it in items if it.image is not None)]
    tile_h = int(tile_w * float(np.clip(np.median(aspects), 0.3, 0.8) if aspects else 0.6)) + 50
    per = cols * rows
    n_sheets = max(1, math.ceil(len(items) / per))
    sheets = []
    for k in range(n_sheets):
        chunk = items[k * per : (k + 1) * per]
        n_rows = max(1, math.ceil(len(chunk) / cols))
        head = caption_bar(
            width, [f"review: {batch}  sheet {k + 1}/{n_sheets}  ({len(items)} frames)", LEGEND], 0.55, (70, 45, 20)
        )
        body = np.full((gap + n_rows * (tile_h + gap), width, 3), 16, np.uint8)
        for i, it in enumerate(chunk):
            r, c = divmod(i, cols)
            y, x = gap + r * (tile_h + gap), gap + c * (tile_w + gap)
            body[y : y + tile_h, x : x + tile_w] = render_tile(it, tile_w, tile_h, warn_px)
        sheets.append(np.vstack([head, body]))
    return sheets


# ---------------------------------------------------------------------------------------------------------------
# Batch


def run_review(
    label_dir: Path,
    out_root: Path,
    batch: str,
    bases: list[Path] | None = None,
    frames_dir: Path | None = None,
    cols: int = 3,
    rows: int = 3,
    width: int = 2000,
    zoom: float = 3.0,
    quality: int = 85,
    warn_px: float = 3.0,
) -> dict:
    """Render all review images for a batch. Returns {"out": dir, "items": [stats], "files": [paths]}."""
    out = Path(out_root) / batch
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    items = load_batch(Path(label_dir), bases or [Path.cwd()], frames_dir)
    if not items:
        raise ValueError(f"no annotated frames in {label_dir}")
    jpg = [cv2.IMWRITE_JPEG_QUALITY, quality]
    files = []
    for it in items:
        p = out / "frames" / f"{it.name}.jpg"
        cv2.imwrite(str(p), render_full(it, warn_px), jpg)
        files.append(p)
        strip = neck_strip(it, zoom=zoom, max_width=width)
        if strip is not None:
            p = out / "strips" / f"{it.name}.jpg"
            cv2.imwrite(str(p), strip, jpg)
            files.append(p)
    for k, sheet in enumerate(contact_sheets(items, batch, cols, rows, width, warn_px)):
        p = out / f"contact_sheet_{k + 1:02d}.jpg"
        cv2.imwrite(str(p), sheet, jpg)
        files.append(p)
    base = (bases or [Path.cwd()])[0].resolve()

    def rel(src: str) -> str:
        p = Path(src.split("@")[0])
        return src.replace(str(p), str(p.relative_to(base))) if p.is_absolute() and p.is_relative_to(base) else src

    stats = [it.stats() | {"image": rel(it.image_src)} for it in items]
    p = out / "summary.csv"
    with p.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0]))
        wr.writeheader()
        for s in stats:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()})
    files.append(p)
    return {"out": out, "items": stats, "files": files}
