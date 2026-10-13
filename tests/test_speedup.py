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


def test_tracking_homography_preserves_slots_and_confidence():
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from speedup_tracking import propagate

    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (160, 320, 3), dtype=np.uint8)
    transform = np.float32([[1, 0, 3], [0, 1, -2]])
    current = cv2.warpAffine(previous, transform, (320, 160))
    k = np.array([[20 + 10 * (i // 2), 60 + 40 * (i % 2), 0.9] for i in range(50)], dtype=np.float32)
    actual, ok = propagate(previous, current, k, 21)
    assert ok
    np.testing.assert_allclose(actual[:, :2], k[:, :2] + [3, -2], atol=0.5)
    np.testing.assert_array_equal(actual[:, 2], k[:, 2])
    unchanged, ok = propagate(np.zeros_like(previous), np.zeros_like(current), k, 21)
    assert not ok
    np.testing.assert_array_equal(unchanged, k)
