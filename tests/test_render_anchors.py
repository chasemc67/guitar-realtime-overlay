import json

import numpy as np

from gro import anchors as A
from gro.geometry import dlt, fret_u, model_from_h, plane_point
from gro.render import mirror_frame, render_on_original, render_overlay, with_mirror_toggle
from gro.theory import scale_positions


def model():
    src = plane_point(np.array([0, 0, 0.7, 0.7]), np.array([0, 1, 0, 1]))
    dst = np.array([[600, 100], [620, 160], [40, 260], [70, 380]], float)
    return model_from_h(dlt(src, dst))


def test_render_draws_and_mirror_round_trip():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (400, 640, 3), dtype=np.uint8)
    m = model()
    notes = scale_positions("A", "minor_pentatonic")
    out = render_overlay(frame, m, notes)
    assert out.shape == frame.shape and (out != frame).any()

    # Toggle path: the clip is "left-handed", i.e. the canonical frame is the mirrored one.
    # Rendering on the original with M.H must equal mirror(render on the canonical frame) — without text labels,
    # which are intentionally kept un-mirrored.
    original = mirror_frame(frame)
    a = render_on_original(original, m, mirrored=True, notes=notes, style=_nolabel())
    b = with_mirror_toggle(original, True, lambda f: render_overlay(f, m, notes, style=_nolabel()))
    diff = np.abs(a.astype(int) - b.astype(int))
    # Anti-aliasing / sub-pixel rounding at polygon edges differs by a pixel at most in a few places.
    assert (diff > 40).mean() < 0.01


def _nolabel():
    from gro.render import Style

    return Style(label="none")


def test_anchor_schema_round_trip(tmp_path):
    m = model()
    anchors = []
    for n in [0, 1, 3, 5]:
        for side, y in (("bass", 0.0), ("treble", 1.0)):
            p = m.project(fret_u(n), y)
            anchors.append(A.Anchor(n, side, float(p[0]), float(p[1]), 2))
    lf = A.LabelFile(
        "clip",
        {"path": "x.mp4", "w": 640, "h": 400, "fps": 60, "mirrored": True},
        {"type": "electric", "frets": 24},
        [A.FrameLabel(t=1.0, anchors=anchors)],
    )
    p = tmp_path / "a.json"
    A.save(lf, p)
    d = json.loads(p.read_text())
    assert d["schema"] == A.SCHEMA
    assert d["frames"][0]["kpts"]["nut"]["bass"][2] == 2
    lf2 = A.load(p)
    assert lf2.mirrored
    assert sorted(lf2.frames[0].anchors, key=lambda a: (a.fret, a.side)) == sorted(
        [A.Anchor(a.fret, a.side, round(a.x, 2), round(a.y, 2), a.vis) for a in anchors],
        key=lambda a: (a.fret, a.side),
    )
    assert A.slot_fret("f12") == 12 and A.slot_name(0) == "nut"
