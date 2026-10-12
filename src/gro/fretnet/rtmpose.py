"""RTMPose-style top-down fretboard keypoints (round 2b), pure PyTorch: no MMPose / mmcv code.

- Backbone: CSPNeXt (RTMDet / RTMPose's backbone), re-implemented here with MMDet's parameter names so the OpenMMLab
  ImageNet checkpoint (`cspnext-m_8xb256-rsb-a1-600e_in1k`, Apache-2.0 model zoo; see LICENSES.md) loads strictly.
- Head: RTMCC (RTMPose): 7x7 conv to one map per keypoint, flatten, linear to a token per keypoint, a gated attention
  unit (GAU) across the 50 keypoint tokens, then SimCC 1D classification of x and y (split 2). Plus a confidence logit
  per token (RTMPose has none; we need "is this wire there").
- Top-down: the input is an affine crop of the source frame around the neck box (W x H = 768 x 384 by default, box
  expanded 1.25x and to that aspect). Frames are 1280x720, so the crop has about twice the pixels per fret of the 640
  letterbox. Train boxes come from the labels (jittered); at inference the box comes from a first-stage model
  (`TopDown`: any FretNet checkpoint, keypoints with conf > 0.5).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_KPTS = 50
CSPNEXT = {"m": (0.67, 0.75), "s": (0.33, 0.5), "l": (1.0, 1.0)}  # deepen, widen


class ConvModule(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=True)


class DWSep(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    def __init__(self, c: int, identity: bool):
        super().__init__()
        self.conv1 = ConvModule(c, c, 3)
        self.conv2 = DWSep(c, c, 5)
        self.identity = identity

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.identity else y


class ChannelAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * F.hardsigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CSPLayer(nn.Module):
    def __init__(self, cin: int, cout: int, n: int, identity: bool):
        super().__init__()
        h = cout // 2
        self.main_conv = ConvModule(cin, h)
        self.short_conv = ConvModule(cin, h)
        self.final_conv = ConvModule(2 * h, cout)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(h, identity) for _ in range(n)])
        self.attention = ChannelAttention(2 * h)

    def forward(self, x):
        y = torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1)
        return self.final_conv(self.attention(y))


class SPPF(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = ConvModule(cin, cin // 2)
        self.conv2 = ConvModule(cin // 2 * 4, cout)

    def forward(self, x):
        x = self.conv1(x)
        y1 = F.max_pool2d(x, 5, 1, 2)
        y2 = F.max_pool2d(y1, 5, 1, 2)
        return self.conv2(torch.cat([x, y1, y2, F.max_pool2d(y2, 5, 1, 2)], 1))


class CSPNeXt(nn.Module):
    """P5 CSPNeXt; stage outputs at strides 4, 8, 16, 32 (channels 96..768 for size m)."""

    def __init__(self, size: str = "m"):
        super().__init__()
        dd, wd = CSPNEXT[size]
        w = lambda c: int(c * wd)  # noqa: E731
        self.stem = nn.Sequential(ConvModule(3, w(32), 3, 2), ConvModule(w(32), w(32), 3), ConvModule(w(32), w(64), 3))
        arch = [(64, 128, 3, True, False), (128, 256, 6, True, False), (256, 512, 6, True, False),
                (512, 1024, 3, False, True)]  # fmt: skip
        self.channels = tuple(w(c[1]) for c in arch)
        for i, (cin, cout, n, identity, spp) in enumerate(arch):
            mods = [ConvModule(w(cin), w(cout), 3, 2)]
            if spp:
                mods.append(SPPF(w(cout), w(cout)))
            mods.append(CSPLayer(w(cout), w(cout), max(round(n * dd), 1), identity))
            setattr(self, f"stage{i + 1}", nn.Sequential(*mods))

    def forward(self, x):
        x = self.stem(x)
        out = []
        for i in range(4):
            x = getattr(self, f"stage{i + 1}")(x)
            out.append(x)
        return out


class ScaleNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale, self.g = d**-0.5, nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) * self.scale).clamp(min=1e-5) * self.g


class GAU(nn.Module):
    """RTMPose's gated attention unit over the keypoint tokens (squared-ReLU attention, rotary-free variant)."""

    def __init__(self, d: int = 256, s: int = 128, expansion: int = 2):
        super().__init__()
        e = d * expansion
        self.e, self.s = e, s
        self.ln = ScaleNorm(d)
        self.uv = nn.Linear(d, 2 * e + s)
        self.gamma = nn.Parameter(torch.randn(2, s) * 0.02)
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.o = nn.Linear(e, d)
        self.res_scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        u, v, base = torch.split(F.silu(self.uv(self.ln(x))), [self.e, self.e, self.s], -1)
        base = base.unsqueeze(2) * self.gamma[None, None] + self.beta[None, None]
        q, k = base.unbind(2)
        a = torch.square(F.relu(q @ k.transpose(1, 2) / math.sqrt(self.s)))
        return x * self.res_scale + self.o(u * (a @ v))


class RTMPoseNet(nn.Module):
    """x (B, 3, H, W) RGB in [0, 1] (crop) -> SimCC x logits (B, 50, W*split), y logits (B, 50, H*split), conf (B, 50)."""

    head = "rtmcc"

    def __init__(self, size: str = "m", pretrained: str | None = None, w: int = 768, h: int = 384, split: int = 2,
                 d: int = 256, window: int = 3):  # fmt: skip
        super().__init__()
        self.size, self.w, self.h, self.split, self.window = size, w, h, split, window
        self.arch = f"cspnext_{size}"
        self.backbone = CSPNeXt(size)
        if pretrained:
            sd = torch.load(pretrained, map_location="cpu", weights_only=False)["state_dict"]
            sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
            self.backbone.load_state_dict(sd, strict=True)
        c = self.backbone.channels[3]
        self.final_layer = nn.Conv2d(c, N_KPTS, 7, padding=3)
        cells = (w // 32) * (h // 32)
        self.mlp = nn.Sequential(ScaleNorm(cells), nn.Linear(cells, d, bias=False))
        self.gau = GAU(d)
        self.cls_x = nn.Linear(d, w * split, bias=False)
        self.cls_y = nn.Linear(d, h * split, bias=False)
        self.conf = nn.Linear(d, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        f = self.backbone((x - self.mean) / self.std)[3]
        tok = self.gau(self.mlp(self.final_layer(f).flatten(2)))
        return self.cls_x(tok), self.cls_y(tok), self.conf(tok)[..., 0]

    def locate1d(self, logits: torch.Tensor) -> torch.Tensor:
        n = logits.shape[-1]
        g = torch.arange(n, device=logits.device, dtype=logits.dtype).view(1, 1, -1)
        peak = (logits >= logits.amax(-1, keepdim=True)).to(logits.dtype)
        i = ((peak * g).sum(-1, keepdim=True) / peak.sum(-1, keepdim=True)).round()
        p = torch.softmax(logits.masked_fill((g - i).abs() > self.window, -1e4), -1)
        return ((p * g).sum(-1) + 0.5) / self.split - 0.5

    def predict(self, x):
        sx, sy, conf = self(x)
        return torch.stack([self.locate1d(sx.float()), self.locate1d(sy.float())], -1), conf.float()


def save(net: RTMPoseNet, path, **meta) -> None:
    torch.save({"kind": "rtmpose", "size": net.size, "w": net.w, "h": net.h, "split": net.split,
                "state_dict": net.state_dict(), "meta": meta}, path)  # fmt: skip


def load(path, device: str = "cpu") -> RTMPoseNet:
    ck = torch.load(path, map_location=device, weights_only=True)
    net = RTMPoseNet(ck["size"], None, ck["w"], ck["h"], ck["split"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


def box_affine(box: np.ndarray, w: int, h: int, expand: float = 1.25, angle: float = 0.0) -> np.ndarray:
    """box (x0, y0, x1, y1) in source px -> 2x3 affine source -> (w, h) crop. The box is expanded by `expand` and to the
    crop's aspect (centred), optionally rotated by `angle` degrees about its centre."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = max(box[2] - box[0], 8.0) * expand, max(box[3] - box[1], 8.0) * expand
    if bw / bh > w / h:
        bh = bw * h / w
    else:
        bw = bh * w / h
    s = w / bw
    a = math.radians(angle)
    c, sn = math.cos(a) * s, math.sin(a) * s
    m = np.array([[c, -sn, 0], [sn, c, 0]], np.float32)
    m[:, 2] = np.array([w / 2, h / 2]) - m[:, :2] @ np.array([cx, cy])
    return m


def kpt_box(k: np.ndarray, vis: np.ndarray) -> np.ndarray | None:
    if vis.sum() < 2:
        return None
    p = k[vis]
    return np.array([*p.min(0), *p.max(0)], np.float32)


class TopDown:
    """Two-stage inference: a FretNet checkpoint (stage 1, full frame) gives the neck box, the RTMPose net (stage 2)
    places the 50 keypoints in the crop. `predict_frame(BGR image)` -> (50, 3) x, y, conf in source px, or None."""

    def __init__(self, stage1, stage2, device: str = "cuda", expand: float = 1.25, thr: float = 0.5):
        from gro.infer import load_backend

        self.s1 = load_backend(stage1, device=device)
        self.s2 = load(stage2, device)
        self.device, self.expand, self.thr = device, expand, thr
        self.imgsz = self.s1.imgsz

    def box(self, image: np.ndarray):
        from gro.infer import Letterbox, decode

        lb = Letterbox.for_shape(image.shape, self.s1.imgsz)
        k, _, _ = decode(self.s1(cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB)), 0.25)
        if not np.isfinite(k[:, 0]).all():
            return None
        k[:, :2] = lb.to_frame(k[:, :2])
        return kpt_box(k[:, :2], k[:, 2] > self.thr)

    @torch.no_grad()
    def predict_frame(self, image: np.ndarray):
        b = self.box(image)
        if b is None:
            return None
        n = self.s2
        m = box_affine(b, n.w, n.h, self.expand)
        crop = cv2.warpAffine(image, m, (n.w, n.h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        x = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None]
        x = x.to(self.device).float() / 255
        xy, conf = n.predict(x)
        xy = xy[0].cpu().numpy()
        mi = cv2.invertAffineTransform(m)
        src = xy @ mi[:, :2].T + mi[:, 2]
        return np.concatenate([src, torch.sigmoid(conf[0]).cpu().numpy()[:, None]], 1)
