"""Train the right-handed 50-keypoint pose model, then export best.pt.

Requires ultralytics (and TensorRT for --trt). First run scripts/export_yolo.py
if data/datasets/scale-v1/data.yaml is absent; training never creates the dataset.
Example: python scripts/train_yolo.py --name scale-v1 [--trt]
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
    a = ap.parse_args()
    if Path(a.name).name != a.name or a.name in (".", ".."):
        ap.error("--name must be a single directory name")
    data = ROOT / "data/datasets/scale-v1/data.yaml"
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
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.2,
        translate=0.1,
        degrees=5.0,
        perspective=0.0005,
    )
    best = YOLO(str(model.trainer.best), task="pose")
    print(best.export(format="onnx", imgsz=a.imgsz, batch=1, dynamic=False))
    if a.trt:
        print(best.export(format="engine", imgsz=a.imgsz, batch=1, half=True))


if __name__ == "__main__":
    main()
