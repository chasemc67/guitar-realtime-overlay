import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from gro.fretnet.model import STRIDE, FretNet, export_head, is_checkpoint, load, save  # noqa: E402
from gro.infer import FretNetBackend, decode, load_backend  # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return FretNet("resnet18", pretrained=False).eval()


def test_locate_recovers_a_peak(net):
    hm = torch.full((1, 50, 160, 160), -10.0)
    hm[0, 3, 40, 100] = 5.0  # cell (x=100, y=40) -> centre px (100.5 * 4 - 0.5, 40.5 * 4 - 0.5)
    hm[0, 3, 40, 101] = 5.0  # equal neighbour: halfway
    hm[0, 3, 120, 10] = 4.9  # a far, lower peak must not pull the location
    xy = net.locate(hm)[0, 3]
    assert xy[0].item() == pytest.approx(101.0 * STRIDE - 0.5, abs=1e-3)
    assert xy[1].item() == pytest.approx(40.5 * STRIDE - 0.5, abs=1e-3)


def test_export_head_matches_v1_raw_layout(net):
    out = export_head(net)(torch.rand(1, 3, 640, 640))
    assert out.shape == (1, 155, 1)
    raw = out[0].detach().numpy()
    k, score, box = decode(raw, conf_thr=0.0)
    conf = raw[5:].reshape(50, 3)[:, 2]
    assert score == pytest.approx(conf.max())
    assert np.all((conf >= 0) & (conf <= 1))
    assert box[0] == pytest.approx((k[:, 0].min() + k[:, 0].max()) / 2, abs=1e-3)


def test_checkpoint_roundtrip_and_backend(net, tmp_path):
    p = tmp_path / "v3.pt"
    save(net, p, epoch=1)
    assert is_checkpoint(p)
    assert not is_checkpoint(tmp_path / "missing.pt")
    again = load(p)
    x = torch.rand(1, 3, 640, 640)
    assert torch.allclose(export_head(again)(x), export_head(net)(x), atol=1e-5)
    backend = load_backend(p, device="cpu")
    assert isinstance(backend, FretNetBackend)
    raw = backend(np.full((640, 640, 3), 114, np.uint8))
    assert raw.shape == (155, 1) and raw.dtype == np.float32


def test_order_gate_drops_slips_and_keeps_a_clean_neck():
    from gro.fretnet.model import R, order_gate

    xs = [20 + 600 * (1 - R**f) for f in range(25)]
    xy = torch.zeros(1, 50, 2)
    for f in range(23):
        for s in (0, 1):
            xy[0, 2 * f + s] = torch.tensor([xs[f], 100.0 + 50 * s])
    conf = torch.zeros(1, 50)
    conf[0, :46] = 0.9
    assert torch.equal(order_gate(xy, conf), conf)
    xy[0, 2 * 17] = xy[0, 2 * 18].clone()  # fret 17 (low-E side) slipped onto wire 18
    xy[0, 2 * 5 + 1] = xy[0, 2 * 12 + 1].clone()  # fret 5 (high-E side) jumped to wire 12
    dropped = (order_gate(xy, conf)[0, :46] < 0.5).nonzero().flatten().tolist()
    assert dropped == [11, 34]


def test_gate_flag_roundtrips(net, tmp_path):
    p = tmp_path / "g.pt"
    net.gate = True
    save(net, p)
    net.gate = False
    assert load(p).gate and export_head(load(p)).gate


def test_wire_gate_drops_one_sided_slips():
    from gro.fretnet.model import R, order_gate

    xs = [20 + 600 * (1 - R**f) for f in range(25)]
    xy = torch.zeros(1, 50, 2)
    for f in range(23):  # slanted wires: the treble end leans forward a little more up the neck
        xy[0, 2 * f] = torch.tensor([xs[f], 100.0])
        xy[0, 2 * f + 1] = torch.tensor([xs[f] + 0.1 * f, 150.0])
    conf = torch.zeros(1, 50)
    conf[0, :46] = 0.9
    assert torch.equal(order_gate(xy, conf, wires=True), conf)
    for f in (19, 20, 21):  # bass ends of 19..21 one wire too far up the neck (a self-consistent run)
        xy[0, 2 * f, 0] = xs[f + 1]
    dropped = (order_gate(xy, conf, wires=True)[0, :46] < 0.5).nonzero().flatten().tolist()
    # 21b duplicates wire 22, so the order gate drops it and 21t (correct) has no pair to check; 19 and 20 lose both
    assert dropped == [38, 39, 40, 41, 42]
