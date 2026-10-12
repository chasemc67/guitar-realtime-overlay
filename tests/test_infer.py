import numpy as np
import pytest

from gro.geometry import model_from_h
from gro.infer import Backend, FretboardTracker, Letterbox, decode, render_state
from gro.infer.tracker import mirror_kpts

W, H = 1280, 720


def _canon_model():
    # canonical plane -> a 1280 x 720 frame, neck roughly horizontal, slight perspective
    return model_from_h(np.array([[1500.0, 30, 80], [10, 220, 330], [0.1, 0, 1]]))


def _kpts(model):
    k = np.zeros((50, 3), np.float32)
    for i in range(50):
        k[i, :2] = model.project_fy(i // 2, float(i % 2))
        k[i, 2] = 0.9
    return k


class FakeBackend(Backend):
    """Returns a raw head output whose best anchor carries the given canonical-frame keypoints."""

    def __init__(self, k_canon, shape=(H, W), size=640, anchors=8400):
        self.imgsz, self.k, self.lb, self.anchors = size, k_canon, Letterbox.for_shape(shape, size), anchors
        self.calls = 0

    def __call__(self, rgb):
        assert rgb.shape == (self.imgsz, self.imgsz, 3) and rgb.dtype == np.uint8
        self.calls += 1
        raw = np.zeros((155, self.anchors), np.float32)
        raw[4] = 0.01
        raw[4, 123] = 0.9
        k = self.k.copy()
        k[:, 0] = k[:, 0] * self.lb.new_w / self.lb.w + self.lb.left
        k[:, 1] = k[:, 1] * self.lb.new_h / self.lb.h + self.lb.top
        raw[5:, 123] = k.ravel()
        return raw


def test_letterbox_matches_ultralytics_rounding_and_roundtrips():
    lb = Letterbox.for_shape((1080, 1920), 640)
    assert (lb.new_w, lb.new_h, lb.left, lb.top) == (640, 360, 0, 140)
    img = lb.apply(np.zeros((1080, 1920, 3), np.uint8))
    assert img.shape == (640, 640, 3) and img[0, 0, 0] == 114 and img[320, 320, 0] == 0
    pts = np.array([[0.0, 0.0], [1919.0, 1079.0], [800.5, 400.25]])
    fwd = np.stack([pts[:, 0] * lb.new_w / lb.w + lb.left, pts[:, 1] * lb.new_h / lb.h + lb.top], 1)
    np.testing.assert_allclose(lb.to_frame(fwd), pts, atol=1e-3)
    assert lb.to_frame(np.array([[-50.0, 700.0]]))[0].tolist() == [0.0, 1080.0]  # clipped like scale_coords
    assert lb.to_frame(np.array([[-50.0, 700.0]]), clip=False)[0, 0] < 0


def test_decode_best_anchor_and_threshold():
    raw = np.zeros((1, 155, 10), np.float32)
    raw[0, 4] = [0.1, 0.8, 0.3, 0, 0, 0, 0, 0, 0, 0]
    raw[0, 5:, 1] = np.arange(150)
    k, score, box = decode(raw)
    assert score == pytest.approx(0.8) and k.shape == (50, 3) and k[1].tolist() == [3, 4, 5]
    k, score, _ = decode(raw, conf_thr=0.9)
    assert np.isnan(k).all() and score == pytest.approx(0.8)


def test_mirror_kpts_is_involution():
    k = np.array([[10.0, 5.0, 0.9], [100.0, 7.0, 0.1]], np.float32)
    np.testing.assert_allclose(mirror_kpts(mirror_kpts(k, W), W), k)
    assert mirror_kpts(k, W)[0, 0] == W - 1 - 10


@pytest.mark.parametrize("mirrored", [False, True])
def test_tracker_recovers_fit_in_original_coords(mirrored):
    canon = _canon_model()
    tracker = FretboardTracker(FakeBackend(_kpts(canon)), mirrored=mirrored, frets=24, smooth=False)
    frame = np.full((H, W, 3), 60, np.uint8)
    state = tracker.process(frame, 0.0)
    assert state.valid and not state.held and state.n_frets == 24
    for n in (0, 5, 12, 24):
        expect = canon.fret_line(n)
        if mirrored:
            expect[:, 0] = W - 1 - expect[:, 0]
        np.testing.assert_allclose(state.model.fret_line(n), expect, atol=0.5)
        np.testing.assert_allclose(state.kpts[2 * n, :2], expect[0], atol=0.5)
    assert state.wires().shape == (25, 2, 2)
    assert set(state.timings_ms) >= {"model", "post", "total"}
    out = render_state(frame, state, show_kpts=True)
    assert out.shape == frame.shape and (out != frame).any()


def test_tracker_holds_then_drops_and_smooths_static_input():
    canon = _canon_model()
    backend = FakeBackend(_kpts(canon))
    tracker = FretboardTracker(backend, frets=24, hold=2, fps=30)
    frame = np.full((H, W, 3), 60, np.uint8)
    for i in range(3):
        s = tracker.process(frame, i / 30)
    np.testing.assert_allclose(s.model.fret_line(12), canon.fret_line(12), atol=0.5)  # static -> filter converges
    backend.k = np.full((50, 3), np.nan, np.float32)  # board disappears (score stays high, kpts NaN)
    held = [tracker.process(frame, (3 + i) / 30) for i in range(3)]
    assert [s.valid for s in held] == [True, True, False] and held[0].held


def test_tracker_without_fret_override_votes_default():
    tracker = FretboardTracker(FakeBackend(_kpts(_canon_model())), smooth=False)
    s = tracker.process(np.full((H, W, 3), 60, np.uint8))  # flat frame: no wire ridges -> no estimate
    assert s.valid and s.n_frets == 24 and s.end_estimate is None
