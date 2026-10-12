"""FretNet: a permissively licensed 50-keypoint fretboard model (model v3). No Ultralytics code or weights.

Backbone: torchvision ResNet (BSD-3-Clause code, ImageNet weights from torchvision; see LICENSES.md). Head (ours): an
FPN that merges strides 4..32 into one stride-4 map, one heatmap per keypoint, and a per-keypoint confidence. Input is
the same 640x640 letterboxed RGB frame as v1, in [0, 1] (the ImageNet normalisation is inside the network), so every
pre-processing path (Python, Swift) is shared with v1.

Keypoint k = 2 * fret + side (side 0 = low-E edge, 1 = high-E edge), frets 0 (nut) .. 24, as in v1.

Location = soft-argmax of the heatmap inside a (2r+1)^2 window around its argmax (frets are periodic, so a global
soft-argmax would average neighbouring wires). `export_head` wraps the network so it emits v1's raw head layout
(1, 155, 1): [cx, cy, w, h, score, x0, y0, c0, ..., x49, y49, c49] in letterboxed px, i.e. one "anchor". Every v1
decoder (gro.infer.decode, Swift FretCore) then reads it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_KPTS = 50
STRIDE = 4
ARCHS = {
    # name: (torchvision constructor, ImageNet weights enum name, channels of layer1..layer4)
    "resnet18": ("resnet18", "ResNet18_Weights", (64, 128, 256, 512)),
    "resnet34": ("resnet34", "ResNet34_Weights", (64, 128, 256, 512)),
    "resnet50": ("resnet50", "ResNet50_Weights", (256, 512, 1024, 2048)),
}


def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class FretNet(nn.Module):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True, ch: int = 128, window: int = 2):
        super().__init__()
        import torchvision

        ctor, wname, chans = ARCHS[arch]
        weights = getattr(torchvision.models, wname).DEFAULT if pretrained else None
        r = getattr(torchvision.models, ctor)(weights=weights)
        self.arch, self.window = arch, window
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.lateral = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in chans])
        self.fuse = nn.Sequential(conv_bn(ch, ch), conv_bn(ch, ch))
        self.heatmap = nn.Conv2d(ch, N_KPTS, 1)
        self.conf = nn.Sequential(nn.Linear(chans[-1] + N_KPTS, 256), nn.ReLU(inplace=True), nn.Linear(256, N_KPTS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem((x - self.mean) / self.std)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        p = self.lateral[3](feats[3])
        for i in (2, 1, 0):
            p = self.lateral[i](feats[i]) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        return self.fuse(p), feats[3]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, 3, S, S) RGB in [0, 1] -> (heatmap logits (B, 50, S/4, S/4), confidence logits (B, 50))."""
        f, c5 = self.features(x)
        hm = self.heatmap(f)
        pooled = torch.cat([c5.mean((2, 3)), hm.flatten(2).amax(-1)], 1)
        return hm, self.conf(pooled)

    def locate(self, hm: torch.Tensor) -> torch.Tensor:
        """Heatmap logits (B, K, H, W) -> (B, K, 2) x, y in input px: soft-argmax in a window around the argmax."""
        b, k, h, w = hm.shape
        flat = hm.flatten(2)
        idx = flat.argmax(-1, keepdim=True)  # (B, K, 1)
        iy = torch.div(idx, w, rounding_mode="floor").to(hm.dtype)
        ix = (idx - torch.div(idx, w, rounding_mode="floor") * w).to(hm.dtype)
        gx = torch.arange(w, device=hm.device, dtype=hm.dtype).repeat(h).view(1, 1, -1)
        gy = torch.arange(h, device=hm.device, dtype=hm.dtype).repeat_interleave(w).view(1, 1, -1)
        inside = ((gx - ix).abs() <= self.window) & ((gy - iy).abs() <= self.window)
        p = torch.softmax(flat.masked_fill(~inside, -1e4), -1)
        x = (p * gx).sum(-1)
        y = (p * gy).sum(-1)
        return (torch.stack([x, y], -1) + 0.5) * STRIDE - 0.5


class ExportHead(nn.Module):
    """FretNet -> v1 raw head layout (1, 155, 1): box from keypoint extent, score = max keypoint confidence."""

    def __init__(self, net: FretNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hm, conf_logit = self.net(x)
        xy = self.net.locate(hm)  # (B, K, 2)
        conf = torch.sigmoid(conf_logit)  # (B, K)
        lo, hi = xy.amin(1), xy.amax(1)  # (B, 2)
        box = torch.cat([(lo + hi) / 2, hi - lo], 1)  # cx, cy, w, h
        score = conf.amax(1, keepdim=True)
        kpts = torch.cat([xy, conf.unsqueeze(-1)], -1).flatten(1)  # (B, 150)
        return torch.cat([box, score, kpts], 1).unsqueeze(-1)  # (B, 155, 1)


def export_head(net: FretNet) -> ExportHead:
    return ExportHead(net).eval()


def save(net: FretNet, path, **meta) -> None:
    """Checkpoint of plain tensors and strings only, so torch.load(weights_only=True) can read it."""
    torch.save({"arch": net.arch, "window": net.window, "state_dict": net.state_dict(), "meta": meta}, path)


def load(path, device: str = "cpu") -> FretNet:
    ck = torch.load(path, map_location=device, weights_only=True)
    net = FretNet(ck["arch"], pretrained=False, window=ck.get("window", 2))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


def is_checkpoint(path) -> bool:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # ultralytics checkpoints pickle their classes and fail weights_only loading
        return False
    return isinstance(ck, dict) and "arch" in ck and "state_dict" in ck
