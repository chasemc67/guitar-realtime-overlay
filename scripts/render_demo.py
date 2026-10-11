"""M1 demo: fit the fretboard model to hand-placed anchors and render a scale overlay.

Stills:  samples/out/<frame>_overlay.jpg (and <frame>_debug.jpg with anchors + fitted wires).
Video:   samples/out/demo.mp4 (gitignored) + samples/out/demo.gif / demo_strip.jpg (committed). The anchors' fit on
         the annotated frame is propagated over ~3 s with pyramidal Lucas-Kanade: Shi-Tomasi corners inside the
         fretboard polygon, a RANSAC inter-frame homography composed onto H (PLAN §2.4). Optional --resnap refines
         each frame by snapping predicted wires to the image ridge and refitting.

    uv run python scripts/render_demo.py [--key A --scale minor_pentatonic] [--video NAME] [--seconds 3] [--resnap]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.geometry import FitError, FretboardModel, fit_fretboard, transform_model
from gro.render import draw_anchors, mirror_frame, render_on_original
from gro.snap import snap_wire
from gro.theory import scale_positions

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ROOT / "samples" / "anchors"
FRAMES = ROOT / "samples" / "frames"
OUT = ROOT / "samples" / "out"


def fit_label(lf: A.LabelFile) -> FretboardModel:
    anchors = lf.frames[0].present()
    n_frets = int(lf.guitar.get("frets", 24))
    # Taper is only observable with many frets spread along the neck; otherwise use the default.
    tau = None if len({a.fret for a in anchors}) >= 8 else 0.3
    return fit_fretboard(anchors, tau=tau, n_frets=n_frets)


def fitted_range(lf: A.LabelFile, model: FretboardModel, shape) -> tuple[int, int]:
    """Draw from the first annotated cell to the last visible fret (D4: every fret in view)."""
    frets = [a.fret for a in lf.frames[0].present()]
    h, w = shape[:2]
    vis = model.visible_frets(w, h)
    lo = max(1, min(frets))
    hi = min(model.n_frets, max(max(frets), max(vis) if vis else 0))
    return lo, hi


def debug_image(img, lf, canon_model, mirrored):
    from gro.geometry import mirror_model

    w = img.shape[1]
    model = mirror_model(canon_model, w) if mirrored else canon_model
    anchors = lf.frames[0].present()
    raw = A.mirror_anchors(anchors, w) if mirrored else anchors
    out = img.copy()
    for n in range(0, model.n_frets + 1):
        a, b = model.fret_line(n)
        cv2.line(out, tuple(np.int32(np.round(a))), tuple(np.int32(np.round(b))), (0, 255, 0), 1, cv2.LINE_AA)
    out = draw_anchors(out, raw, (0, 0, 255))
    return out


def render_still(name: str, notes, args) -> tuple[A.LabelFile, FretboardModel]:
    lf = A.load(ANCHORS / f"{name}.json")
    img = cv2.imread(str(FRAMES / f"{name}.jpg"))
    model = fit_label(lf)
    rng = fitted_range(lf, model, img.shape)
    out = render_on_original(img, model, lf.mirrored, notes=notes, fret_range=rng)
    cv2.imwrite(str(OUT / f"{name}_overlay.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(
        str(OUT / f"{name}_debug.jpg"),
        debug_image(img, lf, model, lf.mirrored),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    print(
        f"{name}: tau={model.tau:.2f} rmse={model.rmse_px:.2f}px inliers={len(model.inliers)}/"
        f"{len(lf.frames[0].present())} frets {rng[0]}-{rng[1]}"
    )
    return lf, model


# ---------------------------------------------------------------------------------------------------------------
# Video: LK propagation in canonical space


def board_mask(model: FretboardModel, shape, f_lo: int, f_hi: int, shrink: float = 0.06) -> np.ndarray:
    from gro.geometry import fret_u

    u0, u1 = fret_u(max(0, f_lo - 1)), fret_u(f_hi)
    poly = model.project(np.array([u0, u1, u1, u0]), np.array([shrink, shrink, 1 - shrink, 1 - shrink]))
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 255)
    return m


def resnap(gray: np.ndarray, model: FretboardModel, frets, min_score: float = 6.0) -> FretboardModel:
    """Snap each predicted wire to the nearest bright ridge and refit (keeps LK from drifting)."""
    anchors = []
    for n in frets:
        if n == 0:
            continue
        b, t = model.project_fy(np.array([n, n]), np.array([0.0, 1.0]))
        seg, score = snap_wire(gray, b, t, search=3.0, step=0.5)
        if score < min_score:
            continue
        d = seg[1] - seg[0]
        # Keep the across-neck extent from the model; only the wire line is measured.
        for side, p in (("bass", b), ("treble", t)):
            # Project the predicted end onto the snapped wire line.
            q = seg[0] + ((p - seg[0]) @ d) / (d @ d) * d
            anchors.append(A.Anchor(n, side, float(q[0]), float(q[1])))
    if len({a.fret for a in anchors}) < 4:
        return model
    try:
        m = fit_fretboard(anchors, tau=model.tau, n_frets=model.n_frets)
    except FitError:
        return model
    return m if m.rmse_px < 2.5 else model


def wire_offsets(gray: np.ndarray, model: FretboardModel, frets, search: float = 6.0) -> np.ndarray:
    """Alignment check: for each predicted wire with a clear ridge, the distance (px) from the predicted wire to the
    best-matching ridge within ±search px (mean of both ends). Small = overlay sits on the real wires."""
    out = []
    for n in frets:
        b, t = model.project_fy(np.array([n, n]), np.array([0.15, 0.85]))
        seg, score = snap_wire(gray, b, t, search=search, step=0.5)
        if score >= 8.0:
            out.append(0.5 * (np.linalg.norm(seg[0] - b) + np.linalg.norm(seg[1] - t)))
    return np.asarray(out)


def render_video(lf: A.LabelFile, model0: FretboardModel, notes, args) -> list[np.ndarray]:
    path = ROOT / lf.video["path"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    t0 = float(lf.frames[0].t)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {path}")
    ref = cv2.imread(str(FRAMES / f"{lf.clip_id}.jpg"))
    if ref is not None and ref.shape == frame.shape:
        print(f"annotated frame vs video @ {t0}s: mean abs diff {np.abs(ref.astype(int) - frame).mean():.1f}")
    n_frames = int(args.seconds * fps)
    h, w = frame.shape[:2]
    mir = lf.mirrored
    canon = mirror_frame(frame) if mir else frame
    prev = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    model = model0
    f_lo, f_hi = fitted_range(lf, model0, frame.shape)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT / "demo.mp4"), fourcc, fps, (w, h))
    frames_out, motion = [], []
    pts = None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for i in range(n_frames):
        if i > 0:
            ok, frame = cap.read()
            if not ok:
                break
            canon = mirror_frame(frame) if mir else frame
            gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
            if pts is None or len(pts) < 40:
                pts = cv2.goodFeaturesToTrack(
                    prev, 300, 0.01, 6, mask=board_mask(model, prev.shape, f_lo, f_hi), blockSize=5
                )
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **lk)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **lk)
                fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
                if good.sum() >= 8:
                    G, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 1.5)
                    if G is not None:
                        model = transform_model(model, G)
                        keep = np.flatnonzero(good)[inl.ravel() == 1]
                        pts = nxt[keep].reshape(-1, 1, 2)
                    else:
                        pts = None
                else:
                    pts = None
            if args.resnap:
                model = resnap(gray, model, range(f_lo, f_hi + 1))
            prev = gray
        if i % int(fps / 2) == 0 or i == n_frames - 1:
            g = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
            off = wire_offsets(cv2.GaussianBlur(g, (3, 3), 0), model, range(max(1, f_lo), f_hi + 1))
            print(
                f"  t=+{i / fps:4.2f}s wire offset: median {np.median(off):.2f}px, p90 {np.percentile(off, 90):.2f}px "
                f"({len(off)} wires with a clear ridge)"
            )
        out = render_on_original(frame, model, mir, notes=notes, fret_range=(f_lo, f_hi))
        writer.write(out)
        frames_out.append(out)
        motion.append(np.linalg.norm(model.project_fy(12, 0.5) - model0.project_fy(12, 0.5)))
    writer.release()
    print(f"wrote {OUT / 'demo.mp4'} ({len(frames_out)} frames); fret-12 centre moved up to {max(motion):.1f}px")
    return frames_out


def save_gif(frames: list[np.ndarray], path: Path, crop=None, width: int = 560, step: int = 4, fps: float = 15):
    """Small animated GIF via ffmpeg palettegen (kept <= ~5 MB so it can be committed)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames[::step]):
            if crop is not None:
                x0, y0, x1, y1 = crop
                f = f[y0:y1, x0:x1]
            f = cv2.resize(f, (width, int(round(f.shape[0] * width / f.shape[1]))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{td}/{i:04d}.png", f)
        cmd = (
            f"ffmpeg -y -v error -framerate {fps} -i {td}/%04d.png -vf "
            f'"split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" {path}'
        )
        subprocess.run(cmd, shell=True, check=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="A")
    ap.add_argument("--scale", default="minor_pentatonic")
    ap.add_argument("--video", default="2026-09-06_09-00-01_10s", help="annotated frame to propagate ('' = none)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--resnap", action="store_true")
    ap.add_argument("--gif-crop", type=int, nargs=4, default=[150, 200, 1070, 540], help="x0 y0 x1 y1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    notes = scale_positions(args.key, args.scale, max_fret=24)
    labels = {}
    for p in sorted(ANCHORS.glob("*.json")):
        labels[p.stem] = render_still(p.stem, notes, args)
    if args.video:
        lf, model = labels[args.video]
        frames = render_video(lf, model, notes, args)
        save_gif(frames, OUT / "demo.gif", crop=args.gif_crop)
        strip = [frames[i] for i in (0, len(frames) // 2, len(frames) - 1)]
        cv2.imwrite(str(OUT / "demo_strip.jpg"), np.vstack(strip), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
