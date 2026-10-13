"""Export v4q-832 using the standard FretNet exporter with trace-stable attention.

Pass the same arguments as export_fretnet.py, including --imgsz 832 --coreml PATH.
The VAL-selected geometric refit is native postprocessing, not baked into the network.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from export_fretnet import main

if __name__ == "__main__":
    # Transformer eval fastpaths otherwise produce different graphs during trace checking.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(4)
    main()
    if "--coreml" in sys.argv:
        import coremltools as ct

        path = Path(sys.argv[sys.argv.index("--coreml") + 1])
        model = ct.models.MLModel(str(path), skip_model_load=True)
        model.short_description = "FretNet v4q-832 fretboard keypoints; native 0.30-fret refit postprocessing"
        model.user_defined_metadata["gro.refit"] = (
            "robust fit (100 samples, seed 0); reassign; reject distance >=0.30 fret"
        )
        # coremltools removes an existing destination before copying the package;
        # never save a loaded package onto its own source directory.
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            destination = Path(directory) / path.name
            model.save(str(destination))
            shutil.rmtree(path)
            shutil.move(str(destination), str(path))
