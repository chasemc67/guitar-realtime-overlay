"""Batch-1 ONNX Runtime CUDA timing for exported pose graphs (needs only numpy + onnxruntime-gpu).

    python scripts/bench_onnx.py runs/eval_v1/best.onnx [runs/eval_v1/best_fp16.onnx ...]

Same protocol as eval_model.py: resident CUDA I/O binding, 20 warmups + 200 iterations, forward only. Use an
onnxruntime-gpu build matching the installed CUDA (the PC's main venv has CUDA 12 wheels; ORT 1.31 needs CUDA 13,
so the v1 numbers came from a separate venv with onnxruntime-gpu 1.22 + nvidia-*-cu12).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import onnxruntime as ort


def bench(path: str, iters: int = 200, warmup: int = 20) -> dict:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        return {"unavailable": "CUDA provider failed to initialize"}
    inp = session.get_inputs()[0]
    dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
    x = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=dtype)
    binding = session.io_binding()
    binding.bind_ortvalue_input(inp.name, ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
    for output in session.get_outputs():
        binding.bind_output(output.name, "cuda", 0)
    for _ in range(warmup):
        session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    start = time.perf_counter()
    for _ in range(iters):
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
    seconds = time.perf_counter() - start
    return {"fps": iters / seconds, "ms": seconds * 1000 / iters, "dtype": np.dtype(dtype).name, "ort": ort.__version__}


if __name__ == "__main__":
    print(json.dumps({p: bench(p) for p in sys.argv[1:]}, indent=2))
