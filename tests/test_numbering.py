"""Geometry, offset sign and safe fallback for the two-stage numbering path."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from gro.fretnet.numbering import NumberingRefiner, rectify, renumber  # noqa: E402
from gro.geometry import FretboardModel, fret_u  # noqa: E402


def test_renumber_preserves_wire_coordinates_and_sides():
    k = np.arange(150, dtype=np.float32).reshape(50, 3)
    for offset in (-3, 0, 3):
        out = renumber(k, offset)
        for fret in range(25):
            if 0 <= fret + offset < 25:
                np.testing.assert_array_equal(
                    out[2 * (fret + offset) : 2 * (fret + offset) + 2], k[2 * fret : 2 * fret + 2]
                )
        if offset:
            empty = out[: 2 * offset] if offset > 0 else out[2 * offset :]
            assert np.all(empty == 0)


def test_rectify_runs_from_nut_to_body():
    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0)
    image = np.zeros((128, 320, 3), np.uint8)
    image[:, :, 0] = np.arange(320).clip(0, 255)
    out = rectify(image, model, width=25, lo=0, hi=24, height=3)
    np.testing.assert_allclose(out[1, :, 0], 40 + 250 * fret_u(np.arange(25)), atol=1)


def test_low_confidence_and_degenerate_fit_fall_back():
    refiner = object.__new__(NumberingRefiner)
    refiner.threshold = 0.8
    refiner.reassign = False
    refiner.net = type("Net", (), {"width": 128})()
    refiner.classify = lambda strip: (2, 0.7)
    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    k = np.zeros((50, 3), np.float32)
    for f in range(25):
        k[2 * f : 2 * f + 2, :2] = model.fret_line(f)
    k[:, 2] = 0.9
    image = np.zeros((128, 320, 3), np.uint8)
    np.testing.assert_array_equal(refiner(image, k), k)
    assert refiner.last["fit_ok"] and not refiner.last["applied"]
    refiner.reassign = True
    refiner.reject_ambiguous = False
    k[10, :2] = model.fret_line(6)[0]
    out = refiner(image, k)
    assert out[10, 2] == 0 and not refiner.last["applied"]
    np.testing.assert_array_equal(out[12], k[10])
    k[10, :2] = model.fret_line(5)[0]
    refiner.reassign = False
    refiner.classify = lambda strip: (-2, 0.9)
    np.testing.assert_array_equal(refiner(image, k), renumber(k, -2))
    k[:, 2] = 0
    np.testing.assert_array_equal(refiner(image, k), k)
    assert not refiner.last["fit_ok"]


def test_reassignment_corrects_isolated_slip_without_moving_pixels():
    from gro.fretnet.numbering import reassign_wires

    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    k = np.zeros((50, 3), np.float32)
    k[10, :2] = model.fret_line(6)[0]
    k[10, 2] = 0.9
    out = reassign_wires(k, model)
    np.testing.assert_array_equal(out[12], k[10])
    assert out[10, 2] == 0
    out = reassign_wires(k, model, offset=-1)
    np.testing.assert_array_equal(out[10], k[10])


def test_training_guard_excludes_clip_ids_and_honest_date(tmp_path):
    import runpy
    from pathlib import Path

    guard = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_numbering.py"))["guarded_images"]
    directory = tmp_path / "images/train"
    directory.mkdir(parents=True)
    regular = directory / "yt_train_000906.jpg"
    regular.touch()
    cfg = {"clips": [{"id": "yt_heldout"}]}
    assert guard(tmp_path, "train", cfg) == [regular]
    for name in ["yt_heldout_000000_wrap.jpg", "2026-09-06_09-00-01_000000.jpg"]:
        bad = directory / name
        bad.touch()
        with pytest.raises(ValueError, match="held-out"):
            guard(tmp_path, "train", cfg)
        bad.unlink()


def test_sequence_head_and_per_wire_fallback():
    from gro.fretnet.numbering import NumberingNet, offset_summary, sequence_renumber

    net = NumberingNet(width=128, mode="sequence").eval()
    with torch.inference_mode():
        assert net(torch.zeros(1, 3, 128, 128)).shape == (1, 25, 25)
    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    k = np.zeros((50, 3), np.float32)
    k[10, :2] = model.fret_line(6)[0]
    k[10, 2] = 0.9
    p = np.eye(25)
    assert offset_summary(p) == (0, 1.0)
    np.testing.assert_array_equal(sequence_renumber(k, model, p, 0.8)[12], k[10])
    p[6] = 1 / 25
    np.testing.assert_array_equal(sequence_renumber(k, model, p, 0.8)[10], k[10])
    np.testing.assert_array_equal(sequence_renumber(k, model, p, 0.8, reassign=True)[12], k[10])
    k[10, :2] = model.fret_line(5.49)[0]
    assert not sequence_renumber(k, model, p, 0.8, reject_ambiguous=True)[:, 2].any()


def test_reject_ambiguous_only_drops_between_wire_points():
    from gro.fretnet.numbering import reassign_wires

    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    k = np.zeros((50, 3), np.float32)
    k[10, :2] = model.fret_line(5.49)[0]
    k[10, 2] = 0.9
    k[12, :2] = model.fret_line(6.1)[0]
    k[12, 2] = 0.9
    assert reassign_wires(k, model)[10, 2] == pytest.approx(0.9)
    out = reassign_wires(k, model, reject_ambiguous=True)
    assert out[10, 2] == 0
    np.testing.assert_array_equal(out[12], k[12])


@pytest.mark.parametrize("mode", ["offset", "sequence"])
def test_mobile_checkpoint_roundtrip(mode, tmp_path):
    from gro.fretnet.numbering import NumberingNet, load_numbering

    net = NumberingNet(width=128, mode=mode, arch="mobilenet_v3_small").eval()
    checkpoint = tmp_path / "numbering.pt"
    torch.save(
        {
            "kind": "gro_numbering",
            "width": 128,
            "mode": mode,
            "arch": "mobilenet_v3_small",
            "state_dict": net.state_dict(),
        },
        checkpoint,
    )
    loaded = load_numbering(checkpoint)
    x = torch.zeros(1, 3, 128, 128)
    with torch.inference_mode():
        expected = net(x)
        assert expected.shape == ((1, 7) if mode == "offset" else (1, 25, 25))
        torch.testing.assert_close(loaded(x), expected)


def test_sequence_dataset_exposes_only_labelled_wires(tmp_path):
    import runpy
    from pathlib import Path

    import cv2

    dataset = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_numbering.py"))["Strips"]
    image = tmp_path / "train_frame.jpg"
    cv2.imwrite(str(image), np.zeros((160, 1729, 3), np.uint8))
    label = np.zeros((50, 3))
    label[6:8, 2] = 2
    label[14, 2] = 1
    (tmp_path / "train_frame.txt").write_text("0 .5 .5 1 1 " + " ".join(map(str, label.ravel())))
    data = dataset([image], 128, False, tmp_path)
    x, shift, present = data[3]
    assert x.shape == (3, 128, 128) and shift == 3
    assert np.flatnonzero(present).tolist() == [3, 7]


def test_source_resolution_preserves_fit_and_coordinates():
    from gro.infer import FretboardTracker, Letterbox, TwoStageBackend

    image = np.full((128, 320, 3), (10, 20, 30), np.uint8)
    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    expected = np.zeros((50, 3), np.float32)
    for f in range(25):
        expected[2 * f : 2 * f + 2, :2] = model.fret_line(f)
    expected[:, 2] = 0.9
    lb = Letterbox.for_shape(image.shape, 128)
    k = expected.copy()
    k[:, 0] = k[:, 0] * lb.new_w / lb.w + lb.left
    k[:, 1] = k[:, 1] * lb.new_h / lb.h + lb.top
    raw = np.zeros((155, 1), np.float32)
    raw[4] = 0.9
    raw[5:, 0] = k.ravel()

    class Refiner:
        def __call__(self, rgb, points, fit_model=None):
            assert rgb.shape == image.shape
            np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
            np.testing.assert_allclose(fit_model.fret_line(12), model.fret_line(12), atol=1e-3)
            return points

    backend = object.__new__(TwoStageBackend)
    backend.source_resolution, backend.imgsz = True, 128
    backend.stage1 = lambda rgb: raw.copy()
    backend.refiner = Refiner()
    tracker = FretboardTracker(backend, smooth=False)
    out, score = tracker.infer(image)
    assert score == pytest.approx(0.9)
    np.testing.assert_allclose(out, expected, atol=2e-5)
    np.testing.assert_array_equal(backend.last_k, k)
