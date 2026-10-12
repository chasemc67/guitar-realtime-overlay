import cv2
import numpy as np

from gro.board_end import FretCountVote, frame_last_fret, ridge_scores
from gro.geometry import fret_u, model_from_h
from gro.smooth import OneEuroModel


def _board_model(n_frets=24):
    # canonical (u in [0, ~0.75], y in [0, 1]) -> a 1600 x 400 image, neck horizontal
    return model_from_h(np.array([[2000.0, 0, 50], [0, 200, 100], [0, 0, 1]]), n_frets=n_frets)


def _synthetic(last_fret):
    """Dark board with bright wires 1..last_fret, nothing beyond."""
    m = _board_model()
    img = np.full((400, 1700), 40, np.uint8)
    for n in range(1, last_fret + 1):
        (x0, y0), (x1, y1) = m.fret_line(n)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 220, 2)
    return img, m


def test_ridges_find_last_fret():
    for last in (21, 22, 24):
        img, m = _synthetic(last)
        r = ridge_scores(img, m)
        assert frame_last_fret(r) == last, (last, r)


def test_frame_last_fret_out_of_view_is_none():
    assert frame_last_fret({19: 1.0, 20: 1.0, 21: 1.0}) is None  # 22 not in view: can't tell 21 from more
    assert frame_last_fret({19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0}) == 21


def test_vote_hysteresis_and_override():
    v = FretCountVote(min_votes=5, margin=3)
    for _ in range(4):
        assert v.update(22) == 24  # not enough votes yet
    assert v.update(22) == 22
    for _ in range(5):
        v.update(24)
    assert v.value == 22  # 5 vs 5: no switch without the margin
    assert FretCountVote(override=21).update(24) == 21


def test_one_euro_static_input_is_identity_and_noise_is_damped():
    m = _board_model(22)
    f = OneEuroModel(30.0)
    rng = np.random.default_rng(0)
    u, y = fret_u(np.array([0, 0, 12, 12, 22, 22.0])), np.array([0, 1, 0, 1, 0, 1.0])
    out_dev, in_dev = [], []
    for _ in range(60):
        H = m.H.copy()
        H[:2, 2] += rng.normal(0, 2.0, 2)  # 2 px translation jitter
        noisy = model_from_h(H, n_frets=22)
        s = f.update(noisy, (400, 1700))
        in_dev.append(np.abs(noisy.project(u, y) - m.project(u, y)).mean())
        out_dev.append(np.abs(s.project(u, y) - m.project(u, y)).mean())
    assert np.mean(out_dev[20:]) < 0.5 * np.mean(in_dev[20:])


def test_startup_cap_until_vote_settles():
    v = FretCountVote(min_votes=5, margin=3)
    conf = np.full(25, 0.95)
    conf[23:] = 0.2  # 22-fret guitar: slots 23/24 unconfident
    assert v.shown == 24  # no data yet
    v.observe_conf(conf)
    assert v.cap == 22 and v.shown == 22
    spurious = conf.copy()
    spurious[24] = 0.9
    v.observe_conf(spurious)  # a single spurious high slot must not raise the cap
    for _ in range(3):
        v.observe_conf(conf)
    assert v.shown == 22
    for _ in range(5):
        v.update(24)  # 24-fret guitar votes: settled, the vote wins over the cap
    assert v.settled and v.shown == 24
    assert FretCountVote(override=21).shown == 21
