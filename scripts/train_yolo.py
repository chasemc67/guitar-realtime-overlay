"""Train the right-handed 50-keypoint pose model, then export best.pt.

Requires ultralytics (and TensorRT for --trt). First run scripts/export_yolo.py
if data/datasets/scale-v1/data.yaml is absent; training never creates the dataset.
Example: python scripts/train_yolo.py --name scale-v1 [--trt]
         python scripts/train_yolo.py --name v2b --data data/datasets/scale-v2/data.yaml --degrees 15 --scale 0.5
Augmentation defaults are v1's; fliplr stays 0 (the model is right-handed only, D1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="scale-v1")
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1/data.yaml")
    ap.add_argument("--degrees", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--translate", type=float, default=0.1)
    ap.add_argument("--perspective", type=float, default=0.0005)
    ap.add_argument("--hsv-s", type=float, default=0.3)
    ap.add_argument("--hsv-v", type=float, default=0.2)
    ap.add_argument("--erasing", type=float, default=0.4)
    ap.add_argument("--close-mosaic", type=int, default=10)
    a = ap.parse_args()
    if Path(a.name).name != a.name or a.name in (".", ".."):
        ap.error("--name must be a single directory name")
    data = a.data.resolve()
    if not data.is_file():
        ap.error(f"missing {data}; run scripts/export_yolo.py first")
    if (ROOT / "runs/pose" / a.name).exists():
        ap.error("run directory already exists; choose a new --name")
    from ultralytics import YOLO

    model = YOLO(a.model, task="pose")
    model.train(
        data=str(data),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        workers=a.workers,
        project=str(ROOT / "runs/pose"),
        name=a.name,
        exist_ok=False,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=a.hsv_s,
        hsv_v=a.hsv_v,
        scale=a.scale,
        translate=a.translate,
        degrees=a.degrees,
        perspective=a.perspective,
        erasing=a.erasing,
        close_mosaic=a.close_mosaic,
    )
    best = YOLO(str(model.trainer.best), task="pose")
    print(best.export(format="onnx", imgsz=a.imgsz, batch=1, dynamic=False))
    if a.trt:
        print(best.export(format="engine", imgsz=a.imgsz, batch=1, half=True))


if __name__ == "__main__":
    main()
