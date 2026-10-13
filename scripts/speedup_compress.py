"""CoreML PTQ candidates, with activation calibration exclusively from VAL."""

# ruff: noqa: E402
import argparse
import json
import sys
import traceback
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as opt
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gro.infer import Letterbox


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--kind", choices=["int8", "palette6", "palette4", "w8a8"], required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("Refusing to overwrite model")
    report = dict(source=str(a.model), kind=a.kind, coremltools=ct.__version__)
    try:
        model = ct.models.MLModel(str(a.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
        if a.kind == "int8":
            result = opt.linear_quantize_weights(
                model, opt.OptimizationConfig(global_config=opt.OpLinearQuantizerConfig(dtype="int8"))
            )
        elif a.kind.startswith("palette"):
            result = opt.palettize_weights(
                model, opt.OptimizationConfig(global_config=opt.OpPalettizerConfig(nbits=int(a.kind[-1])))
            )
        else:
            assert a.bundle is not None
            desc = model.get_spec().description.input[0]
            size = desc.type.imageType.width
            data = []
            for row in json.loads((a.bundle / "calibration.json").read_text()):
                image = cv2.imread(str(a.bundle / row["image"]))
                rgb = cv2.cvtColor(Letterbox.for_shape(image.shape, size).apply(image), cv2.COLOR_BGR2RGB)
                data.append({desc.name: Image.fromarray(rgb)})
            config = opt.OptimizationConfig(global_config=opt.OpLinearQuantizerConfig(dtype="int8"))
            result = opt.linear_quantize_activations(model, config, sample_data=data, calibration_op_group_size=32)
            result = opt.linear_quantize_weights(result, config)
        result.save(str(a.out))
        report["status"] = "exported"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    a.out.with_suffix(".compression.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
