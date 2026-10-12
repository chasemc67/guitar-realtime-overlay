"""FretNet: a permissively licensed 50-keypoint fretboard model (model v3). No Ultralytics code or weights.

Backbone: torchvision ResNet (BSD-3-Clause code, ImageNet weights from torchvision; see LICENSES.md). Head (ours): an
FPN that merges strides 4..32 into one stride-4 map, one heatmap per keypoint, and one confidence map per keypoint, read
at the predicted location (local evidence only, like a detector's per-anchor keypoint confidence). Input is
the same 640x640 letterboxed RGB frame as v1, in [0, 1] (the ImageNet normalisation is inside the network), so every
pre-processing path (Python, Swift) is shared with v1.

Keypoint k = 2 * fret + side (side 0 = low-E edge, 1 = high-E edge), frets 0 (nut) .. 24, as in v1.

Location = soft-argmax of the heatmap inside a (2r+1)^2 window around its argmax (frets are periodic, so a global
soft-argmax would average neighbouring wires). `export_head` wraps the network so it emits v1's raw head layout
(1, 155, 1): [cx, cy, w, h, score, x0, y0, c0, ..., x49, y49, c49] in letterboxed px, i.e. one "anchor". Every v1
decoder (gro.infer.decode, Swift FretCore) then reads it unchanged.

Global-context variants (round 2, numbering is a global decision: inlay pattern, nut, body joint, fret count):
- head="query": an encoder (self-attention over the stride-32 tokens, sine positions) feeds the FPN top, and 50
  keypoint queries (a transformer decoder: self-attention among the keypoints, cross-attention to the image tokens)
  add a per-keypoint dynamic heatmap term <q_k, F + pos> and a global confidence term. The local heatmap still sets the
  precise location; the queries pick which wire.
- SimCCNet: RTMPose-style SimCC. Each keypoint is a token made from the whole stride-16 map (a fully connected layer
  over all positions), the 50 tokens attend to each other, and each classifies its x and y over the full canvas in
  1D bins (split 2). Every coordinate is a global decision.
Backbones: torchvision ResNet-18/34/50 or ConvNeXt-T (BSD-3-Clause code, ImageNet-1k weights; see LICENSES.md).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_KPTS = 50
STRIDE = 4
ARCHS = {
    # name: (torchvision constructor, ImageNet weights enum name, channels of stages 1..4 at strides 4..32)
    "resnet18": ("resnet18", "ResNet18_Weights", (64, 128, 256, 512)),
    "resnet34": ("resnet34", "ResNet34_Weights", (64, 128, 256, 512)),
    "resnet50": ("resnet50", "ResNet50_Weights", (256, 512, 1024, 2048)),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", (96, 192, 384, 768)),
}


def backbone(arch: str, pretrained: bool) -> tuple[nn.Module, nn.ModuleList, tuple]:
    """(stem, 4 stages at strides 4, 8, 16, 32, their channels) from a torchvision classifier."""
    import torchvision

    ctor, wname, chans = ARCHS[arch]
    weights = getattr(torchvision.models, wname).DEFAULT if pretrained else None
    r = getattr(torchvision.models, ctor)(weights=weights)
    if arch.startswith("convnext"):
        f = r.features
        return f[0], nn.ModuleList([f[1], nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]),
                                    nn.Sequential(f[6], f[7])]), chans  # fmt: skip
    return nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool), nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4]), chans


def sine_pos(c: int, h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1, c, h, w) 2D sine/cosine position code (DETR style), c / 4 frequencies per axis, positions in [0, 1]."""
    n = c // 4
    freq = 1.0 / (100.0 ** (torch.arange(n, device=device, dtype=torch.float32) / n))
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 * torch.pi
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 * torch.pi
    fy = (y[:, None] * freq[None] * 10).T[:, :, None].expand(n, h, w)
    fx = (x[:, None] * freq[None] * 10).T[:, None, :].expand(n, h, w)
    return torch.cat([fy.sin(), fy.cos(), fx.sin(), fx.cos()])[None].to(dtype)


def encoder(d: int, layers: int, heads: int = 8) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class GlobalQuery(nn.Module):
    """Self-attention over the stride-32 tokens (returned as a map for the FPN top) and 50 keypoint queries decoded
    against them. Outputs: refined stride-32 map (B, d, h, w), heatmap kernels (B, 50, ch), global conf (B, 50)."""

    def __init__(self, c5: int, ch: int, d: int = 256, enc: int = 2, dec: int = 3):
        super().__init__()
        self.inp = nn.Conv2d(c5, d, 1)
        self.enc = encoder(d, enc)
        layer = nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, dec)
        self.query = nn.Parameter(torch.randn(N_KPTS, d) * 0.02)
        self.kernel = nn.Linear(d, ch)
        self.gconf = nn.Linear(d, 1)
        nn.init.zeros_(self.gconf.weight)
        nn.init.zeros_(self.gconf.bias)

    def forward(self, c5: torch.Tensor):
        b, _, h, w = c5.shape
        t = self.inp(c5)
        d = t.shape[1]
        t = (t + sine_pos(d, h, w, t.device, t.dtype)).flatten(2).transpose(1, 2)  # (B, hw, d)
        m = self.enc(t)
        q = self.dec(self.query.to(m.dtype).expand(b, -1, -1), m)  # (B, 50, d)
        return m.transpose(1, 2).reshape(b, d, h, w), self.kernel(q), self.gconf(q)[..., 0]


def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class FretNet(nn.Module):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True, ch: int = 128, window: int = 2,
                 head: str = "heatmap", imgsz: int = 640):  # fmt: skip
        super().__init__()
        self.arch, self.window, self.head, self.imgsz = arch, window, head, imgsz
        self.stem, self.layers, chans = backbone(arch, pretrained)
        self.glob = GlobalQuery(chans[3], ch) if head == "query" else None
        if self.glob is not None:
            chans = (*chans[:3], self.glob.inp.out_channels)
            self.pos = nn.Conv2d(32, ch, 1)  # sine position code -> the dynamic heatmap's feature space
        self.lateral = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in chans])
        self.fuse = nn.Sequential(conv_bn(ch, ch), conv_bn(ch, ch))
        self.heatmap = nn.Conv2d(ch, N_KPTS, 1)
        self.conf = nn.Conv2d(ch, N_KPTS, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, x: torch.Tensor):
        x = self.stem((x - self.mean) / self.std)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        glob = None
        if self.glob is not None:
            feats[3], *glob = self.glob(feats[3])
        p = self.lateral[3](feats[3])
        for i in (2, 1, 0):
            p = self.lateral[i](feats[i]) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        return self.fuse(p), glob

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, 3, S, S) RGB in [0, 1] -> (heatmap logits (B, 50, S/4, S/4), confidence logits (B, 50)).

        Confidence logit = the confidence map averaged under softmax(heatmap), i.e. read where the keypoint is (plus,
        for head="query", each keypoint query's global term)."""
        f, glob = self.features(x)
        hm = self.heatmap(f)
        if glob is not None:
            kernel, gconf = glob
            _, c, h, w = f.shape
            g = f + self.pos(sine_pos(32, h, w, f.device, f.dtype))
            hm = hm + torch.einsum("bkc,bchw->bkhw", kernel, g) / c**0.5
        p = torch.softmax(hm.flatten(2).float(), -1).detach()  # the confidence loss must not move the heatmap
        conf = (p * self.conf(f).flatten(2).float()).sum(-1)
        return hm, conf if glob is None else conf + gconf.float()

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x -> (keypoints (B, 50, 2) in input px, confidence logits (B, 50))."""
        hm, conf = self(x)
        return self.locate(hm.float()), conf

    def locate(self, hm: torch.Tensor) -> torch.Tensor:
        """Heatmap logits (B, K, H, W) -> (B, K, 2) x, y in input px: soft-argmax in a window around the argmax."""
        # The peak cell comes from an equality one-hot rather than a flat argmax index: the index (up to 25599) is not
        # exact in fp16, which CoreML uses on the Neural Engine. Every value here stays below 160.
        b, k, h, w = hm.shape
        flat = hm.flatten(2)
        gx = torch.arange(w, device=hm.device, dtype=hm.dtype).repeat(h).view(1, 1, -1)
        gy = torch.arange(h, device=hm.device, dtype=hm.dtype).repeat_interleave(w).view(1, 1, -1)
        peak = (flat >= flat.amax(-1, keepdim=True)).to(hm.dtype)
        n = peak.sum(-1, keepdim=True)
        ix = ((peak * gx).sum(-1, keepdim=True) / n).round()  # ties: their mean
        iy = ((peak * gy).sum(-1, keepdim=True) / n).round()
        inside = ((gx - ix).abs() <= self.window) & ((gy - iy).abs() <= self.window)
        p = torch.softmax(flat.masked_fill(~inside, -1e4), -1)
        x = (p * gx).sum(-1)
        y = (p * gy).sum(-1)
        return (torch.stack([x, y], -1) + 0.5) * STRIDE - 0.5


class SimCCNet(nn.Module):
    """RTMPose-style SimCC: keypoint tokens from the whole stride-16 map, self-attention among the 50 tokens, then
    x / y classification over the canvas in 1D bins of 1 / split px. Location = soft-argmax in a window around each
    axis' argmax (frets are periodic along x and y); confidence = a linear read of the token."""

    head = "simcc"

    def __init__(self, arch: str = "convnext_tiny", pretrained: bool = True, d: int = 256, split: int = 2,
                 window: int = 3, imgsz: int = 640, layers: int = 2):  # fmt: skip
        super().__init__()
        self.arch, self.split, self.window, self.imgsz = arch, split, window, imgsz
        self.stem, self.layers, chans = backbone(arch, pretrained)
        self.lat4 = nn.Conv2d(chans[3], d, 1)
        self.lat3 = nn.Conv2d(chans[2], d, 1)
        self.enc = encoder(d, 1)  # global context over the stride-32 tokens before the merge
        self.mix = nn.Sequential(conv_bn(d, d), nn.Conv2d(d, N_KPTS, 7, padding=3))
        cells = (imgsz // 16) ** 2
        self.token = nn.Sequential(nn.Linear(cells, d), nn.LayerNorm(d))
        self.kpt_pos = nn.Parameter(torch.randn(N_KPTS, d) * 0.02)
        self.attn = encoder(d, layers)
        self.cls_x = nn.Linear(d, imgsz * split)
        self.cls_y = nn.Linear(d, imgsz * split)
        self.conf = nn.Linear(d, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, 3, S, S) -> (x logits (B, 50, S*split), y logits (B, 50, S*split), confidence logits (B, 50))."""
        x = self.stem((x - self.mean) / self.std)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        c5 = self.lat4(feats[3])
        b, d, h, w = c5.shape
        t = (c5 + sine_pos(d, h, w, c5.device, c5.dtype)).flatten(2).transpose(1, 2)
        c5 = self.enc(t).transpose(1, 2).reshape(b, d, h, w)
        p = self.lat3(feats[2]) + F.interpolate(c5, scale_factor=2.0, mode="nearest")
        tok = self.token(self.mix(p).flatten(2)) + self.kpt_pos.to(p.dtype)  # (B, 50, d)
        tok = self.attn(tok)
        return self.cls_x(tok), self.cls_y(tok), self.conf(tok)[..., 0]

    def locate1d(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, K, N) bin logits -> (B, K) px. Peak by equality one-hot (fp16-safe, as FretNet.locate)."""
        n = logits.shape[-1]
        g = torch.arange(n, device=logits.device, dtype=logits.dtype).view(1, 1, -1)
        peak = (logits >= logits.amax(-1, keepdim=True)).to(logits.dtype)
        i = ((peak * g).sum(-1, keepdim=True) / peak.sum(-1, keepdim=True)).round()
        p = torch.softmax(logits.masked_fill((g - i).abs() > self.window, -1e4), -1)
        return ((p * g).sum(-1) + 0.5) / self.split - 0.5

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sx, sy, conf = self(x)
        return torch.stack([self.locate1d(sx.float()), self.locate1d(sy.float())], -1), conf.float()


R = 2 ** (-1 / 12)  # ratio of consecutive fret gaps (equal temperament)


def order_gate(xy: torch.Tensor, conf: torch.Tensor, thr: float = 0.5, tol1: float = 0.25, tol2: float = 0.2,
               wires: bool = False):  # fmt: skip
    """Fret-order consistency: zero the confidence of a keypoint that sits where its index neighbours say it cannot.

    Per side, keypoints above `thr` are projected on the neck axis (t). Test 1: keypoint f's fraction of the span from
    f-1 to f+1 should be 1 / (1 + R) (about 0.51); test 2: its fraction of the span f-2 .. f+2 should be about 0.51 too.
    A keypoint that slipped onto the next wire fails both (fraction ~1 and ~0.75); its correct neighbours pass the test
    that skips it. Keypoints fail when every available test fails (and at least one is available). Perspective changes
    consecutive gap ratios only mildly, so the fractions stay well within the tolerances."""
    out = []
    for side in (0, 1):
        p, c = xy[:, side::2], conf[:, side::2]  # (B, 25, 2), (B, 25)
        v = (c > thr).to(xy.dtype)
        d = (p[:, 1:] - p[:, :-1]) * (v[:, 1:] * v[:, :-1])[..., None]
        u = d.sum(1, keepdim=True)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
        t = (p * u).sum(-1)  # (B, 25)

        def frac(k: int, expect: float, tol: float):
            n = t.shape[1]
            z = torch.zeros_like(t[:, :k])
            lo_t, hi_t = torch.cat([z, t[:, : n - k]], 1), torch.cat([t[:, k:], z], 1)
            lo_v, hi_v = torch.cat([z, v[:, : n - k]], 1), torch.cat([v[:, k:], z], 1)
            have = lo_v * hi_v * v
            span = hi_t - lo_t
            f = (t - lo_t) / torch.where(span.abs() < 1e-3, torch.ones_like(span), span)
            bad = (((f - expect).abs() > tol) | (span <= 1e-3)).to(t.dtype)
            return have, have * bad

        h1, b1 = frac(1, 1 / (1 + R), tol1)
        h2, b2 = frac(2, (1 + R) / (1 + R + R**2 + R**3), tol2)
        fail = (b1 + (1 - h1)) * (b2 + (1 - h2)) * torch.clamp(h1 + h2, max=1)  # every available test failed
        out.append(c * (1 - fail))
    conf = torch.stack(out, -1).flatten(1)  # back to k = 2 * fret + side
    return wire_gate(xy, conf, thr) if wires else conf


def wire_gate(xy: torch.Tensor, conf: torch.Tensor, thr: float = 0.5, tol: float = 0.5, iters: int = 3):
    """Cross-side check: the two ends of a wire must belong to the same wire. Per fret with both ends above `thr`, the
    offset o_f of the treble end from the bass end along the neck axis varies smoothly with f (wires are near-parallel;
    perspective tilts them gradually). A robust weighted line fit o ~ a + b f (reweighted `iters` times, no sort, so it
    exports to CoreML) gives the expected offset; a wire whose ends are more than `tol` fret gaps off has one end on a
    neighbouring wire (a one-sided slip), so both ends are dropped."""
    pb, pt = xy[:, 0::2], xy[:, 1::2]  # (B, 25, 2)
    cb, ct = conf[:, 0::2], conf[:, 1::2]
    vb, vt = (cb > thr).to(xy.dtype), (ct > thr).to(xy.dtype)
    both = vb * vt
    mid = (pb + pt) / 2
    d = (mid[:, 1:] - mid[:, :-1]) * (both[:, 1:] * both[:, :-1])[..., None]
    u = d.sum(1, keepdim=True)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    o = ((pt - pb) * u).sum(-1)  # (B, 25)
    t = (mid * u).sum(-1)
    gm = both[:, 1:] * both[:, :-1]
    g = (t[:, 1:] - t[:, :-1]).abs() * gm  # gap between consecutive wires (both valid)
    z = torch.zeros_like(g[:, :1])
    gl, gr = torch.cat([z, g], 1), torch.cat([g, z], 1)
    ml, mr = torch.cat([z, gm], 1), torch.cat([gm, z], 1)
    gap = (gl + gr) / (ml + mr).clamp(min=1)  # local fret gap at f
    has_gap = ((ml + mr) > 0).to(xy.dtype)
    f = torch.arange(xy.shape[1] // 2, device=xy.device, dtype=xy.dtype).view(1, -1)
    w = both * has_gap
    for _ in range(iters):  # weighted least squares o ~ a + b f, then down-weight wires far from the fit
        sw = w.sum(1, keepdim=True).clamp(min=1e-6)
        mf, mo = (w * f).sum(1, keepdim=True) / sw, (w * o).sum(1, keepdim=True) / sw
        b = (w * (f - mf) * (o - mo)).sum(1, keepdim=True) / (w * (f - mf) ** 2).sum(1, keepdim=True).clamp(min=1e-6)
        r = (o - (mo + b * (f - mf))) / gap.clamp(min=1.0)
        w = both * has_gap * torch.exp(-((r / tol) ** 2))
    bad = both * has_gap * ((r.abs() > tol).to(xy.dtype)) * ((w.sum(1, keepdim=True) > 3).to(xy.dtype))
    keep = (1 - bad).unsqueeze(-1).expand(-1, -1, 2).flatten(1)
    return conf * keep


class ExportHead(nn.Module):
    """FretNet -> v1 raw head layout (1, 155, 1): box from keypoint extent, score = max keypoint confidence.
    gate=True applies `order_gate` to the confidences (the tracker and every decoder see gated keypoints)."""

    def __init__(self, net: nn.Module, gate: bool | str = False):
        super().__init__()
        self.net, self.gate = net, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy, conf_logit = self.net.predict(x)  # (B, K, 2), (B, K)
        conf = torch.sigmoid(conf_logit)  # (B, K)
        if self.gate:
            conf = order_gate(xy, conf, wires=self.gate == "wires")
        lo, hi = xy.amin(1), xy.amax(1)  # (B, 2)
        box = torch.cat([(lo + hi) / 2, hi - lo], 1)  # cx, cy, w, h
        score = conf.amax(1, keepdim=True)
        kpts = torch.cat([xy, conf.unsqueeze(-1)], -1).flatten(1)  # (B, 150)
        return torch.cat([box, score, kpts], 1).unsqueeze(-1)  # (B, 155, 1)


def export_head(net: nn.Module, gate: bool | str | None = None) -> ExportHead:
    """gate=None: the checkpoint's own setting (`net.gate`, saved by `save(..., gate=True)`)."""
    return ExportHead(net, getattr(net, "gate", False) if gate is None else gate).eval()


def build(arch: str, head: str = "heatmap", pretrained: bool = True, imgsz: int = 640, **kw) -> nn.Module:
    if head == "simcc":
        return SimCCNet(arch, pretrained=pretrained, imgsz=imgsz, **kw)
    return FretNet(arch, pretrained=pretrained, head=head, imgsz=imgsz, **kw)


def save(net: nn.Module, path, **meta) -> None:
    """Checkpoint of plain tensors and strings only, so torch.load(weights_only=True) can read it."""
    torch.save({"arch": net.arch, "window": net.window, "head": net.head, "imgsz": net.imgsz,
                "gate": getattr(net, "gate", False), "state_dict": net.state_dict(), "meta": meta}, path)  # fmt: skip


def load(path, device: str = "cpu") -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=True)
    net = build(ck["arch"], ck.get("head", "heatmap"), pretrained=False, imgsz=ck.get("imgsz", 640),
                window=ck.get("window", 2))  # fmt: skip
    net.load_state_dict(ck["state_dict"])
    net.gate = ck.get("gate", False)  # False, True (order gate) or "wires" (order + cross-side wire gate)
    return net.to(device).eval()


def is_checkpoint(path) -> bool:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # ultralytics checkpoints pickle their classes and fail weights_only loading
        return False
    return isinstance(ck, dict) and "arch" in ck and "state_dict" in ck
