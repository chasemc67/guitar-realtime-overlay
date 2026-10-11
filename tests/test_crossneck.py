"""Cross-neck fix: width prior, manifest exclusion, eval metric."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from gro import anchors as A
from gro.auto import strings as S
from gro.auto.manifest import excluded_ids, video_id
from gro.geometry import dlt, fret_u, model_from_h, plane_point

ROOT = Path(__file__).resolve().parents[1]


def test_rho_is_one_for_a_straight_on_physical_board():
    # Plane u in scale lengths, 1000 px per scale length both ways (no foreshortening), strings at the physical span.
    px = 1000.0
    for n in (1, 7, 12, 20):
        span_px = S.span_mm(n) / S.SCALE_MM * px
        su = px  # px per relative u at shift 0
        assert abs(span_px / su * float(S.rho_factor(n, 0)) - 1.0) < 1e-9


def test_rho_factor_accounts_for_the_index_shift():
    # The same image measurements read with a wrong index shift give a different rho (the prior must try shifts).
    assert S.rho_factor(5, 0) != S.rho_factor(5, 3)
    assert list(S.allowed_shifts([3, 4, 10], None, 24)) == list(range(-3, 17))
    assert list(S.allowed_shifts([3, 4, 10], 2)) == [2]


def test_prior_ok_rejects_slivers_and_drift():
    ok = S.prior_ok(np.array([[0.95, 0.9], [0.2, 0.2], [0.95, 0.6], [1.6, 1.5]]))
    assert ok.tolist() == [True, False, False, False]


def test_prior_lines_span_matches_the_prior():
    src = plane_point(np.array([0, 0, 0.72, 0.72]), np.array([0, 1, 0, 1]), tau=0.3)
    dst = np.array([[880, 170], [890, 245], [120, 300], [135, 400]], float)
    m = model_from_h(dlt(src, dst), tau=0.3)
    frets = list(range(2, 20))
    lines = S.prior_lines(m, frets, 0)
    segs = {n: m.fret_line(n) for n in frets}
    rho = S.lines_rho(m, lines, segs, np.array([0]))
    assert np.allclose(rho, S.RHO_NOM, atol=0.05), rho


def test_manifest_excludes_the_acoustic_clip():
    ex = excluded_ids()
    assert ex.get("s7DVJX4PK3U") == "acoustic"
    assert video_id("s7DVJX4PK3U_08") == "s7DVJX4PK3U"
    assert video_id("8-grcbKnbps_08") == "8-grcbKnbps"


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_auto", ROOT / "scripts" / "eval_auto.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crossneck_metric():
    ev = _eval_module()
    gt, pred = [], []
    for f in range(1, 6):
        x = 100.0 * f
        gt += [A.Anchor(f, "bass", x, 100.0, 2), A.Anchor(f, "treble", x, 200.0, 2)]
        # Prediction: a sliver covering the middle half (span 50 instead of 100), shifted index by 0.
        pred += [A.Anchor(f, "bass", x, 125.0, 2), A.Anchor(f, "treble", x, 175.0, 2)]
    cn = ev.crossneck(gt, pred, 0)
    assert np.allclose(cn["span_px"], 50.0) and np.allclose(cn["span_pct"], 50.0)
    assert np.allclose(cn["outer_px"], 25.0) and np.allclose(cn["outer_pct"], 25.0)
    assert fret_u(12) == 0.5
