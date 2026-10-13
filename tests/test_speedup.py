"""Validate new backbone stage boundaries and checkpoint/export compatibility."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from gro.fretnet.model import FretNet, export_head, load, save  # noqa: E402


@pytest.mark.parametrize("arch", ["mobilenet_v3_large", "efficientnet_b0"])
def test_light_backbone_query_checkpoint(arch, tmp_path):
    torch.set_num_threads(2)
    net = FretNet(arch, pretrained=False, head="query", imgsz=64).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        features = net.stem((x - net.mean) / net.std)
        for layer, size in zip(net.layers, [16, 8, 4, 2], strict=True):
            features = layer(features)
            assert features.shape[-2:] == (size, size)
        expected = export_head(net)(x)
        assert expected.shape == (1, 155, 1)
        assert torch.isfinite(expected).all()
        path = tmp_path / "student.pt"
        save(net, path)
        actual = export_head(load(path))(x)
        torch.testing.assert_close(actual, expected)
