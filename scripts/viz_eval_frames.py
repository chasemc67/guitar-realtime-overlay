"""Draw GT (green) vs model predictions (red, fret index) for frames of an eval_model.py report.

    python scripts/viz_eval_frames.py --weights runs/eval_v1/best.pt --out data/out/eval_viz ITEM [ITEM ...]

ITEM is an `item` from eval/model_<name>_report.json: an exported test image stem (pseudo-GT test split) or
`<video_stem>#<frame_index>` (held-out owner video pseudo-GT, read from data/labels/scale-v1). Images are canonical
(mirrored clips flipped, D1), cropped to the board, and labelled with per-frame D13 stats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_model import metrics, predict  # noqa: E402

from gro.anchors import Anchor  # noqa: E402
from gro.anchors import load as load_label  # noqa: E402


def load_item(item: str, dataset: Path, labels: Path):
    if "#" in item:
        stem, fi = item.split("#")
        for path in sorted(labels.glob(f"{stem}_*.json")):
            lf = load_label(path)
            for fr in lf.frames:
                if int(fr.extra.get("frame_index", -1)) == int(fi):
                    cap = cv2.VideoCapture(str(ROOT / lf.video["path"]))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, image = cap.read()
                    cap.release()
                    return (cv2.flip(image, 1) if lf.mirrored else image), fr.present()
        raise KeyError(item)
    image = cv2.imread(str(dataset / "images/test" / f"{item}.jpg"))
    h, w = image.shape[:2]
    values = np.array([float(v) for v in (dataset / "labels/test" / f"{item}.txt").read_text().split()])
    gt = [
        Anchor(i // 2, ("bass", "treble")[i % 2], x * w, y * h, int(v))
        for i, (x, y, v) in enumerate(values[5:].reshape(50, 3))
        if v > 0
    ]
    return image, gt


def draw(image, gt, pred, title):
    out = image.copy()
    for pts, color, dy in ((gt, (0, 220, 0), -8), (pred, (0, 0, 255), 16)):
        by = {}
        for p in pts:
            by.setdefault(p.fret, {})[p.side] = (int(round(p.x)), int(round(p.y)))
        for f, s in by.items():
            if len(s) == 2:
                cv2.line(out, s["bass"], s["treble"], color, 1, cv2.LINE_AA)
            for xy in s.values():
                cv2.circle(out, xy, 3, color, -1, cv2.LINE_AA)
            x, y = s.get("bass", next(iter(s.values())))
            cv2.putText(out, str(f), (x - 6, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    allp = np.array([[p.x, p.y] for p in gt + pred]) if gt or pred else np.zeros((0, 2))
    if len(allp):
        h, w = out.shape[:2]
        x0, y0 = np.maximum(allp.min(0) - 60, 0).astype(int)
        x1, y1 = np.minimum(allp.max(0) + 60, [w, h]).astype(int)
        out = out[y0:y1, x0:x1]
    if out.shape[1] < 900:
        s = 900 / out.shape[1]
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    bar = np.zeros((28, out.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/scale-v1")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels/scale-v1")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights, task="pose")
    a.out.mkdir(parents=True, exist_ok=True)
    for item in a.items:
        image, gt = load_item(item, a.dataset, a.labels)
        pred, detected = predict(model, image, a.imgsz)
        m = metrics(gt, pred, detected)
        d13 = m["wire_perp_px"]
        title = (
            f"{item}  green=GT red=pred  D13 mean {np.mean(d13) if d13 else float('nan'):.1f}px  "
            f"shifts {sorted(set(m['index_shifts']))}  detected={detected}"
        )
        path = a.out / f"{item.replace('#', '_')}.jpg"
        cv2.imwrite(str(path), draw(image, gt, pred, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


if __name__ == "__main__":
    main()
