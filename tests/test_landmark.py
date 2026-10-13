"""Landmark targets, missing-wire snapping, vote direction, and safe fallback."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from gro.fretnet.landmark import (  # noqa: E402
    BINS,
    LANDMARKS,
    LandmarkNet,
    LandmarkRefiner,
    decode_landmarks,
    load_landmark,
    vote_landmarks,
    wire_positions,
)
from gro.fretnet.numbering import HI, LO  # noqa: E402
from gro.geometry import FretboardModel  # noqa: E402


def detection(fret, position, confidence=0.9):
    return dict(fret=fret, position=position, confidence=confidence, presence=confidence)


def test_agreeing_landmarks_outvote_single_wrong_landmark_and_offset():
    detections = [detection(12, 10), detection(7, 5), detection(9, 7), detection(5, 4)]
    wires = [(f, float(f)) for f in range(25)]
    offset, _, applied, votes = vote_landmarks(detections, wires, offset_vote=(-1, 0.99))
    assert offset == 2 and applied and len(votes) == 4
    assert vote_landmarks([detection(12, 13)], wires)[:3] == (-1, 1.0, True)


def test_ties_absence_low_confidence_and_missing_wires_fall_back():
    wires = [(f, float(f)) for f in range(25)]
    assert vote_landmarks([detection(12, 10), detection(7, 6)], wires)[:3] == (0, 0.5, False)
    assert vote_landmarks([detection(12, 12, 0.1)], wires)[:3] == (0, 0.0, False)
    assert vote_landmarks([detection(12, 12)], [(3, 3.0)])[:3] == (0, 0.0, False)
    assert vote_landmarks([], wires, offset_vote=(2, 0.9))[:3] == (2, 1.0, True)


def test_decode_absence_competes_with_localization():
    p = np.zeros((7, BINS + 1))
    for i, f in enumerate(LANDMARKS):
        p[i, int((f - LO) / (HI - LO) * (BINS - 1))] = 0.9
        p[i, -1] = 0.1
    decoded = decode_landmarks(p)
    np.testing.assert_allclose([d["position"] for d in decoded], LANDMARKS)
    assert all(d["confidence"] == pytest.approx(0.9) for d in decoded)
    p[:] = 0
    p[:, -1] = 1
    assert all(d["confidence"] == d["presence"] == 0 for d in decode_landmarks(p))


def test_refiner_preserves_pixels_and_falls_back_on_failed_fit():
    model = FretboardModel(np.array([[250, 0, 40], [0, 60, 60], [0, 0, 1.0]]), tau=0.3)
    k = np.zeros((50, 3), np.float32)
    for f in range(25):
        k[2 * f : 2 * f + 2, :2] = model.fret_line(f)
    k[:, 2] = 0.9
    wires = wire_positions(k, model)
    np.testing.assert_allclose([v for _, v in wires], np.arange(25), atol=1e-5)
    refiner = object.__new__(LandmarkRefiner)
    refiner.net = type("Net", (), {"width": 128})()
    refiner.threshold, refiner.reassign, refiner.reject_ambiguous = 0.8, False, False
    refiner.offset_model = None
    refiner.detect = lambda strip: [detection(12, 10)]
    out = refiner(np.zeros((128, 320, 3), np.uint8), k, model)
    np.testing.assert_array_equal(out[24:26], k[20:22])
    refiner.detect = lambda strip: [detection(12, 10, 0.1)]
    np.testing.assert_array_equal(refiner(np.zeros((128, 320, 3), np.uint8), k, model), k)
    k[:, 2] = 0
    np.testing.assert_array_equal(refiner(np.zeros((128, 320, 3), np.uint8), k), k)
    assert not refiner.last["fit_ok"]


def test_checkpoint_roundtrip(tmp_path):
    net = LandmarkNet(width=128).eval()
    path = tmp_path / "landmark.pt"
    torch.save(dict(kind="gro_landmark", width=128, landmarks=list(LANDMARKS), state_dict=net.state_dict()), path)
    loaded = load_landmark(path)
    with torch.inference_mode():
        x = torch.zeros(1, 3, 128, 128)
        assert net(x).shape == (1, 7, BINS + 1)
        torch.testing.assert_close(net(x), loaded(x))


def test_dataset_absent_targets_and_jitter_are_consistent(tmp_path):
    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_landmark import LandmarkStrips

    # Vertical bright line at GT fret 12 follows the same warp as its heatmap target.
    image = np.zeros((160, 1729, 3), np.uint8)
    image[:, 18 * 48 - 2 : 18 * 48 + 3] = 255
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), image)
    k = np.zeros((50, 3))
    k[24:26] = [0.5, 0.5, 2]
    (tmp_path / "frame.txt").write_text("0 .5 .5 1 1 " + " ".join(map(str, k.ravel())))
    dataset = LandmarkStrips([path], tmp_path, 1024, False)
    x, target = dataset[0]
    assert target[0, -1] == 0 and (target[1:, -1] == 1).all()
    peak = int(x[:, 64].float().mean(0).argmax()) / 1023 * (BINS - 1)
    assert abs(peak - int(target[0].argmax())) < 1
    # Epoch 1 has no erasure at the landmark, and includes projective jitter.
    dataset.train, dataset.epoch = True, 1
    x, target = dataset[0]
    peak = int(x[:, 64].float().mean(0).argmax()) / 1023 * (BINS - 1)
    assert abs(peak - int(target[0].argmax())) < 1
    assert torch.allclose(target.sum(-1), torch.ones(7))


def test_detection_metrics_include_absence_and_localization_errors():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from landmark_eval import aggregate_diagnostics, score_detections

    detections = [detection(f, f) for f in LANDMARKS]
    detections[0]["position"] += 1
    present = np.ones(7, bool)
    present[-1] = False
    scores = score_detections(detections, LANDMARKS, present, 0.8)
    report = aggregate_diagnostics([{"landmark": {"predicted_strip": scores, "fit_ok": True}}])
    assert report["predicted_strip"]["12"]["recall"] == 0
    assert report["predicted_strip"]["7"]["recall"] == 1
    assert report["predicted_strip"]["0"]["precision"] == 0
    assert report["predicted_strip"]["0"]["recall"] is None
