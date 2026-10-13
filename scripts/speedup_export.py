"""Trace-stable CoreML export carrying the actual candidate and frozen refit metadata."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import torch
from export_fretnet import main as export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--coreml", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    a = ap.parse_args()
    if a.coreml.exists():
        ap.error("Refusing to replace an export")
    selected = json.loads(a.selection.read_text())
    assert selected["imgsz"] == a.imgsz
    assert selected["weights_sha256"] == hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    sys.argv = ["export_fretnet.py", str(a.checkpoint), "--imgsz", str(a.imgsz), "--coreml", str(a.coreml)]
    export()
    model = ct.models.MLModel(str(a.coreml), skip_model_load=True)
    model.short_description = f"FretNet speed experiment {a.checkpoint.stem}, {a.imgsz} input"
    model.user_defined_metadata["gro.refit"] = (
        f"100 samples, seed 0; reject distance >= {selected['chosen_threshold']} fret"
    )
    model.user_defined_metadata["gro.weights_sha256"] = selected["weights_sha256"]
    with tempfile.TemporaryDirectory(dir=a.coreml.parent) as directory:
        destination = Path(directory) / a.coreml.name
        model.save(str(destination))
        shutil.rmtree(a.coreml)
        shutil.move(str(destination), str(a.coreml))


if __name__ == "__main__":
    main()
