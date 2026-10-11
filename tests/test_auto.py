import json

import cv2
import numpy as np

from gro.auto import detect as D
from gro.auto.ai import AIClient, parse_json
from gro.geometry import dlt, fret_u, model_from_h, plane_point


def _abc_x(abc, n):
    u = fret_u(np.asarray(n, float))
    return (abc[0] * u + abc[1]) / (abc[2] * u + 1.0)


def test_reindex_is_an_index_shift():
    abc = np.array([900.0, 100.0, 0.08])
    for s in (-5, 3, 11):
        abc2 = D._reindex(abc, s)
        n = np.arange(0, 12)
        assert np.allclose(_abc_x(abc2, n), _abc_x(abc, n + s), atol=1e-6)


def test_grid_fit_recovers_relative_indices_with_gaps_and_outliers():
    abc = np.array([1400.0, 50.0, 0.1])
    true_n = np.arange(3, 22)
    xs = _abc_x(abc, true_n)
    keep = np.ones(len(xs), bool)
    keep[[4, 9]] = False  # two wires under fingers
    xs_obs = np.r_[xs[keep], [xs[2] + 0.5 * (xs[3] - xs[2]), xs[12] + 0.45 * (xs[13] - xs[12])]]  # 2 inlay peaks
    hs = np.r_[np.full(keep.sum(), 20.0), [12.0, 12.0]]
    o = np.argsort(xs_obs)
    g = D.fit_grid_1d(xs_obs[o], hs[o], (xs.min() - 30, xs.max() + 30))
    assert g is not None
    # Relative indices are consistent with the truth up to one constant shift, increasing toward the body.
    rel = sorted(g.peak_idx)
    assert len(rel) == keep.sum()
    shift = true_n[keep][0] - rel[0]
    assert np.allclose(g.x(np.array(rel)), _abc_x(abc, np.array(rel) + shift), atol=0.2)


def test_grid_fit_direction_from_spacing():
    # Wires laid out right-to-left (nut on the right): indices must still increase toward the shrinking spacing.
    abc = np.array([-1200.0, 1500.0, 0.05])
    xs = np.sort(_abc_x(abc, np.arange(1, 20)))
    g = D.fit_grid_1d(xs, np.full(len(xs), 20.0), (xs.min() - 20, xs.max() + 20))
    lo, hi = g.index_range
    assert g.x(lo) > g.x(hi)  # fret 1 is the rightmost


def test_vanishing_point():
    vp = np.array([400.0, -2000.0])
    p0 = np.array([[100.0 + 30 * k, 300.0] for k in range(8)])
    p1 = p0 + 0.05 * (vp - p0)
    v = D.vanishing_point(p0, p1)
    assert np.allclose(v[:2] / v[2], vp, atol=1.0)


def _synthetic_board(seed=0):
    rng = np.random.default_rng(seed)
    h, w = 540, 960
    img = (rng.normal(90, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
    src = plane_point(np.array([0, 0, 0.72, 0.72]), np.array([0, 1, 0, 1]), tau=0.3)
    dst = np.array([[880, 170], [890, 245], [120, 300], [135, 400]], float)
    m = model_from_h(dlt(src, dst), tau=0.3)
    board = m.project(np.array([0, 0.75, 0.75, 0]), np.array([0, 0, 1, 1]))
    cv2.fillPoly(img, [np.int32(np.round(board))], (30, 25, 22))
    for s in range(6):
        y = 0.08 + 0.84 * s / 5
        a, b = m.project_fy(np.array([0.0, 24.0]), np.array([y, y]))
        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (150, 150, 150), 1, cv2.LINE_AA)
    sh = 4
    for n in range(1, 23):
        a, b = m.fret_line(n)
        cv2.line(
            img,
            tuple(np.int32(np.round(a * 2**sh))),
            tuple(np.int32(np.round(b * 2**sh))),
            (225, 225, 225),
            2,
            cv2.LINE_AA,
            sh,
        )
    return cv2.GaussianBlur(img, (3, 3), 0.8), m


def test_detect_synthetic_board():
    img, m = _synthetic_board()
    det = D.detect_frame(img)
    assert det.ok, det.reason
    vis = sorted(n for n, wv in det.wires.items() if wv.vis == 2)
    assert len(vis) >= 15
    # Relative -> true index by the best constant shift; then wires must sit on the true wires.
    best = None
    for s in range(-25, 25):
        errs = []
        for n in vis:
            f = n + s
            if not 1 <= f <= 22:
                errs.append(99)
                continue
            mid = 0.5 * (det.wires[n].seg[0] + det.wires[n].seg[1])
            a, b = m.fret_line(f)
            d = (b - a) / np.linalg.norm(b - a)
            errs.append(abs(d[0] * (mid - a)[1] - d[1] * (mid - a)[0]))
        e = np.median(errs)
        if best is None or e < best[0]:
            best = (e, s)
    assert best[0] < 0.6
    # Treble anchors land on the true treble edge.
    s = best[1]
    tr = [a for a in det.anchors if a.side == "treble" and a.vis == 2 and 1 <= a.fret + s <= 22]
    err = [np.linalg.norm(m.project_fy(a.fret + s, 1.0) - [a.x, a.y]) for a in tr]
    assert np.median(err) < 1.5


def test_parse_json_and_offline_cache(tmp_path):
    assert parse_json('blah ```json\n{"a": 1, "b": [1,2,],}\n``` end') == {"a": 1, "b": [1, 2]}
    assert parse_json("no json") is None
    ai = AIClient(cache_dir=tmp_path, offline=True)
    img = np.zeros((8, 8, 3), np.uint8)
    assert ai.chat_json("t", "p", [img]) is None
    # Seed the cache for exactly this request: it must be served without any network call.
    from gro.auto.ai import encode_image

    key = ai._key_for(ai.model, "t", "\n\np", [encode_image(img)], 4000)
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / f"{key}.json").write_text(json.dumps({"parsed": {"ok": True}}))
    assert ai.chat_json("t", "p", [img]) == {"ok": True}
    assert ai.usage.cached == 1 and ai.usage.calls == 0
