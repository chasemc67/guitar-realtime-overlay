import numpy as np
import pytest

from gro.anchors import Anchor, mirror_anchors
from gro.geometry import (
    FitError,
    apply_h,
    fit_fretboard,
    fret_u,
    mirror_matrix,
    mirror_model,
    model_from_h,
    plane_point,
    resolve_index_shift,
)


def make_h(seed=0):
    """A plausible plane->image homography: neck diagonal across a 1280x720 frame with some perspective."""
    rng = np.random.default_rng(seed)
    src = plane_point(np.array([0, 0, 0.7, 0.7]), np.array([0, 1, 0, 1]))
    dst = np.array([[1100, 120], [1130, 175], [150, 470], [210, 620]], float) + rng.normal(0, 15, (4, 2))
    from gro.geometry import dlt

    return dlt(src, dst)


def synth_anchors(H, frets, noise=0.0, seed=0, tau=0.3):
    rng = np.random.default_rng(seed)
    out = []
    for n in frets:
        for side, y in (("bass", 0.0), ("treble", 1.0)):
            p = apply_h(H, plane_point(fret_u(n), y, tau)) + rng.normal(0, noise, 2)
            out.append(Anchor(n, side, float(p[0]), float(p[1])))
    return out


def test_fret_spacing():
    assert fret_u(0) == 0
    assert fret_u(12) == pytest.approx(0.5)
    assert fret_u(24) == pytest.approx(0.75)
    gaps = np.diff(fret_u(np.arange(0, 25)))
    assert np.allclose(gaps[1:] / gaps[:-1], 2 ** (-1 / 12))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_homography_round_trip(seed):
    H = make_h(seed)
    anchors = synth_anchors(H, [0, 1, 3, 5, 7, 9, 12])
    m = fit_fretboard(anchors)
    assert m.rmse_px < 1e-3
    # Extrapolated frets (not annotated) land where the true model puts them.
    true = model_from_h(H)
    for n in [2, 15, 19, 22]:
        assert np.allclose(m.fret_line(n), true.fret_line(n), atol=1e-2)
    # Image -> plane -> image round-trip.
    pts = plane_point(np.array([0.1, 0.33, 0.6]), np.array([0.2, 0.5, 0.9]))
    img = apply_h(m.H, pts)
    back = apply_h(np.linalg.inv(m.H), img)
    assert np.allclose(back, pts, atol=1e-9)


def test_partial_frets_known_indices_no_nut():
    H = make_h(3)
    anchors = synth_anchors(H, [5, 6, 7, 9], noise=0.3)
    m = fit_fretboard(anchors)
    true = model_from_h(H)
    assert np.linalg.norm(m.fret_line(8) - true.fret_line(8), axis=1).max() < 2.0
    assert np.linalg.norm(m.fret_line(12) - true.fret_line(12), axis=1).max() < 4.0


def test_ransac_rejects_outliers():
    H = make_h(4)
    anchors = synth_anchors(H, [0, 2, 3, 5, 7, 9, 12, 15], noise=0.4)
    bad = anchors[5]
    anchors[5] = Anchor(bad.fret, bad.side, bad.x + 40, bad.y - 25)
    m = fit_fretboard(anchors)
    assert any(a.fret == bad.fret and a.side == bad.side for a in m.outliers)
    true = model_from_h(H)
    assert np.linalg.norm(m.fret_line(10) - true.fret_line(10), axis=1).max() < 1.5


def test_needs_two_edges():
    H = make_h(0)
    anchors = [a for a in synth_anchors(H, [0, 3, 5, 7]) if a.side == "bass"]
    with pytest.raises(FitError):
        fit_fretboard(anchors)


def camera_h(rvec, t, f, w=1280, h=720):
    """Plane->image H of a physically plausible fretboard (648 mm scale, 43 mm nut) seen by a pinhole camera."""
    import cv2

    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return K @ np.c_[R[:, 0], R[:, 1], t] @ np.diag([0.648, 0.043, 1.0])


POSES = [
    ([0.42, -0.55, 0.03], [-0.16, -0.09, 0.76]),
    ([0.46, -0.42, -0.02], [-0.03, -0.02, 0.74]),
    ([0.42, 0.23, 0.0], [-0.28, 0.0, 0.59]),
    ([-0.12, -0.14, 0.0], [-0.29, 0.0, 0.89]),
]


@pytest.mark.parametrize("pose", range(len(POSES)))
@pytest.mark.parametrize("true_first", [3, 6])
def test_index_shift_without_nut(pose, true_first):
    H = camera_h(*POSES[pose], f=0.8 * 1280)
    anchors = synth_anchors(H, range(true_first, true_first + 7), noise=0.3, seed=true_first)
    # Annotator / network numbered the first visible wire as fret 1.
    wrong = [Anchor(a.fret - true_first + 1, a.side, a.x, a.y) for a in anchors]
    # Spacing alone is (almost) shift-invariant: reprojection error can't tell the shifts apart.
    m_geo = resolve_index_shift(wrong, shifts=range(0, 10))
    assert m_geo.index_margin < 1.5
    # The metric-plane cue with a known focal gets within one fret (exact in most poses, see learnings in
    # resolve_index_shift); inlays/landmarks must settle the rest.
    m = resolve_index_shift(wrong, shifts=range(0, 10), image_size=(1280, 720), focal=0.8 * 1280)
    assert abs(m.index_shift - (true_first - 1)) <= 1
    assert {a.fret for a in m.inliers} == set(range(true_first, true_first + 7)) or m.index_shift != true_first - 1


def test_index_shift_metric_cue_mostly_exact():
    hits = 0
    for pose in range(len(POSES)):
        for true_first in (2, 4, 6, 8):
            H = camera_h(*POSES[pose], f=0.8 * 1280)
            anchors = synth_anchors(H, range(true_first, true_first + 7), noise=0.3, seed=pose * 10 + true_first)
            wrong = [Anchor(a.fret - true_first + 1, a.side, a.x, a.y) for a in anchors]
            m = resolve_index_shift(wrong, shifts=range(0, 10), image_size=(1280, 720), focal=0.8 * 1280)
            hits += m.index_shift == true_first - 1
    assert hits >= 13  # of 16


def test_mirror_composition():
    W = 1280
    H = make_h(6)
    anchors = synth_anchors(H, [0, 2, 5, 7, 12])
    m = fit_fretboard(anchors)
    mm = mirror_model(m, W)
    for n in [0, 4, 12, 17]:
        a = m.fret_line(n)
        b = mm.fret_line(n)
        assert np.allclose(b[:, 0], W - 1 - a[:, 0])
        assert np.allclose(b[:, 1], a[:, 1])
    M = mirror_matrix(W)
    assert np.allclose(M @ M, np.eye(3))
    # Fitting directly on mirrored anchors (no side swap) equals M.H_canon.
    m2 = fit_fretboard(mirror_anchors(anchors, W))
    assert np.allclose(m2.fret_line(9), mm.fret_line(9), atol=1e-3)
    # Mirroring flips the orientation (determinant sign); the h-flip augmentation (with bass/treble swap) keeps it.
    assert np.sign(np.linalg.det(m.H)) != np.sign(np.linalg.det(m2.H))
    m3 = fit_fretboard(mirror_anchors(anchors, W, swap_sides=True))
    assert np.sign(np.linalg.det(m.H)) == np.sign(np.linalg.det(m3.H))


def test_taper_is_used_in_fit():
    H = make_h(7)
    anchors = synth_anchors(H, [0, 2, 4, 7, 10, 14, 19], tau=0.5)
    m = fit_fretboard(anchors, tau=0.5)
    assert m.rmse_px < 1e-3
    m_grid = fit_fretboard(anchors, tau=None)
    assert m_grid.tau == pytest.approx(0.5)
