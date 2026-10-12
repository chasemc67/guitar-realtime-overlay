"""Real-time inference for the fretboard pose model (CoreML / ONNX / PyTorch). See docs/INFERENCE.md."""

from gro.infer.backends import Backend, CoreMLBackend, FretNetBackend, OnnxBackend, TorchBackend, load_backend
from gro.infer.preprocess import Letterbox, decode
from gro.infer.render import draw_hud, render_state
from gro.infer.tracker import FretboardTracker, TrackState, anchors_from

__all__ = [
    "Backend",
    "CoreMLBackend",
    "FretNetBackend",
    "FretboardTracker",
    "Letterbox",
    "OnnxBackend",
    "TorchBackend",
    "TrackState",
    "anchors_from",
    "decode",
    "draw_hud",
    "load_backend",
    "render_state",
]
