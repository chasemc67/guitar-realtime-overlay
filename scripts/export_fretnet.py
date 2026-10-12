"""Export a FretNet (model v3) checkpoint to CoreML (.mlpackage) and/or ONNX. No Ultralytics.

    python scripts/export_fretnet.py runs/fretnet/v3_r50/best.pt --coreml v3_fp16.mlpackage [--fp32] [--onnx v3.onnx]

Both exports carry v1's interface, so every v1 consumer (gro.infer, the Swift FretCore) loads them unchanged:
- CoreML input `image`: 640x640 RGB image type with the 1/255 scale baked in (pass a BGRA CVPixelBuffer).
  ONNX input `images`: float32 (1, 3, 640, 640) RGB in [0, 1].
- One float32 output (1, 155, 1): [cx, cy, w, h, score, x0, y0, c0, ..., x49, y49, c49] in letterboxed px, i.e. v1's
  raw head with a single anchor (box = keypoint extent, score = max keypoint confidence).
CoreML needs coremltools (macOS; coremltools 9 wants torch <= 2.7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gro.fretnet.model import export_head, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--coreml", type=Path, default=None)
    ap.add_argument("--fp32", action="store_true", help="CoreML fp32 instead of fp16")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    model = export_head(load(a.checkpoint, "cpu"))
    x = torch.rand(1, 3, a.imgsz, a.imgsz)
    if a.onnx:
        torch.onnx.export(model, x, str(a.onnx), input_names=["images"], output_names=["output0"], opset_version=17,
                          dynamo=False)  # fmt: skip
        print("wrote", a.onnx)
    if a.coreml:
        import coremltools as ct

        traced = torch.jit.trace(model, x)
        ml = ct.convert(
            traced,
            inputs=[ct.ImageType(name="image", shape=x.shape, scale=1 / 255, color_layout=ct.colorlayout.RGB)],
            outputs=[ct.TensorType(name="output")],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32 if a.fp32 else ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        ml.short_description = "FretNet v3 fretboard keypoints (50), v1 raw layout (1, 155, 1)"
        ml.license = "Own code (no Ultralytics); torchvision ResNet ImageNet weights, BSD-3-Clause. See LICENSES.md"
        ml.save(str(a.coreml))
        print("wrote", a.coreml)


if __name__ == "__main__":
    main()
