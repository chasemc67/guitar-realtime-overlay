"""Inference backends. Each is a callable: RGB uint8 (S, S, 3) letterboxed image -> raw head output (155, A) float32.

Only the chosen backend's runtime is imported: coremltools (macOS, .mlpackage), onnxruntime (.onnx), or ultralytics +
torch (.pt: ultralytics v1, or a FretNet v3 checkpoint, which needs only torch + torchvision).
`load_backend(path)` picks by suffix (and, for .pt, by checkpoint content).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

COMPUTE_UNITS = ("ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY")


class Backend:
    name = "base"
    imgsz = 640

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, n: int = 3) -> None:
        x = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        for _ in range(n):
            self(x)

    def latency_ms(self, n: int = 100, warmup: int = 10) -> np.ndarray:
        """Per-call wall time (ms) on a constant input, batch 1."""
        x = np.random.default_rng(0).integers(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.warmup(warmup)
        out = []
        for _ in range(n):
            t = time.perf_counter()
            self(x)
            out.append((time.perf_counter() - t) * 1000)
        return np.array(out)


class CoreMLBackend(Backend):
    """Core ML .mlpackage exported by ultralytics (image input "image", scale 1/255 baked in, RGB)."""

    name = "coreml"

    def __init__(self, path, compute_units: str = "ALL"):
        import coremltools as ct
        from PIL import Image

        self._image = Image
        self.model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
        spec = self.model.get_spec().description
        self.input = spec.input[0].name
        self.output = spec.output[0].name
        self.imgsz = spec.input[0].type.imageType.width or 640

    def __call__(self, rgb):
        return self.model.predict({self.input: self._image.fromarray(rgb)})[self.output][0]


class OnnxBackend(Backend):
    """ONNX graph exported by ultralytics: input (1, 3, S, S) RGB in [0, 1]."""

    name = "onnx"

    def __init__(self, path, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=providers or ort.get_available_providers())
        inp = self.session.get_inputs()[0]
        self.input = inp.name
        self.dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        self.imgsz = inp.shape[-1] if isinstance(inp.shape[-1], int) else 640

    def __call__(self, rgb):
        x = (rgb.transpose(2, 0, 1)[None] / np.float32(255)).astype(self.dtype)
        return self.session.run(None, {self.input: x})[0][0].astype(np.float32)


class TorchBackend(Backend):
    """Ultralytics .pt weights run as a plain nn.Module (no ultralytics pre/post-processing)."""

    name = "torch"

    def __init__(self, path, device: str | None = None, imgsz: int = 640):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch, self.device, self.imgsz = torch, device, imgsz
        self.model = YOLO(str(path), task="pose").model.fuse().eval().to(device)

    def __call__(self, rgb):
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])).to(self.device).float() / 255
            y = self.model(x)
            y = y[0] if isinstance(y, (list, tuple)) else y
            return y[0].float().cpu().numpy()


class FretNetBackend(Backend):
    """FretNet (model v3, gro.fretnet) checkpoint in PyTorch. Emits v1's raw layout as one anchor: (155, 1)."""

    name = "fretnet"

    def __init__(self, path, device: str | None = None, imgsz: int = 640):
        import torch

        from gro.fretnet.model import export_head, load

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch, self.device, self.imgsz = torch, device, imgsz
        self.model = export_head(load(path, device))

    def __call__(self, rgb):
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])).to(self.device).float() / 255
            return self.model(x)[0].float().cpu().numpy()


def load_backend(path, **kw) -> Backend:
    """By suffix: .mlpackage / .mlmodel -> CoreML (kw compute_units), .onnx -> ONNX (providers), .pt -> torch
    (FretNet checkpoint or ultralytics weights). CoreML / ONNX exports of either model share the raw layout."""
    suffix = Path(path).suffix.lower()
    if suffix in (".mlpackage", ".mlmodel"):
        return CoreMLBackend(path, **kw)
    if suffix == ".onnx":
        return OnnxBackend(path, **kw)
    if suffix == ".pt":
        from gro.fretnet.model import is_checkpoint

        return FretNetBackend(path, **kw) if is_checkpoint(path) else TorchBackend(path, **kw)
    raise ValueError(f"unknown model format: {path}")
