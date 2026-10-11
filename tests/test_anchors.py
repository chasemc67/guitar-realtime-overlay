import json

import pytest

from gro import anchors as A


def _frame(frets, occluded=(), **kw):
    anchors = []
    for f in frets:
        vis = A.VIS_OCCLUDED if f in occluded else A.VIS_VISIBLE
        anchors += [A.Anchor(f, "bass", 10.0 * f, 0.0, vis), A.Anchor(f, "treble", 10.0 * f, 50.0, vis)]
    return A.FrameLabel(t=1.0, anchors=anchors, **kw)


def _file(*frames):
    return A.LabelFile("clip", {"w": 640, "h": 400, "mirrored": False}, {"frets": 24}, list(frames))


@pytest.mark.parametrize(
    ("frets", "occluded", "cue"),
    [
        (range(0, 6), (), "nut"),
        (range(0, 16), (), "nut"),  # nut wins over the inlay cues
        (range(8, 16), (), "inlay_9_12_15"),
        (range(2, 16), (), "inlay_9_12_15"),  # 9-12-15 wins over 3-5-7
        (range(1, 8), (), "inlay_3_5_7"),
        (range(1, 8), (5,), None),  # 3-5-7 needs all three visible
        (range(9, 15), (), None),  # 9 and 12 without 15
        (range(1, 15), (5,), None),
        (range(0, 6), (0,), None),  # occluded nut is not a cue
        ((), (), None),
    ],
)
def test_derived_anchor_cue(frets, occluded, cue):
    fr = _frame(frets, occluded)
    assert fr.derived_anchor_cue() == cue
    assert fr.anchorability() == (cue is not None, cue)


def test_one_visible_side_is_enough():
    fr = _frame(range(1, 8))
    fr.anchors = [a for a in fr.anchors if not (a.fret == 5 and a.side == "treble")]
    assert fr.anchorability() == (True, "inlay_3_5_7")


def test_no_neck_is_not_anchorable():
    assert _frame(range(0, 6), neck_present=False).anchorability() == (False, None)


def test_round_trip_writes_derived_fields():
    d = A.to_dict(_file(_frame(range(8, 16))))
    fr = d["frames"][0]
    assert fr["anchorable"] is True and fr["anchor_cue"] == "inlay_9_12_15"
    back = A.from_dict(json.loads(json.dumps(d))).frames[0]
    # Stored values matching the anchors stay derived, so they track later kpt edits.
    assert back.anchorable is None
    back.anchors = [a for a in back.anchors if a.fret != 15]
    assert back.anchorability() == (False, None)


def test_explicit_override_kept():
    d = A.to_dict(_file(_frame(range(8, 16))))
    d["frames"][0].update(anchorable=False, anchor_cue=None)  # e.g. annotator saw a blank board
    fr = A.from_dict(d).frames[0]
    assert fr.anchorability() == (False, None)
    assert A.to_dict(_file(fr))["frames"][0]["anchorable"] is False

    d["frames"][0].update(anchorable=True, anchor_cue="nut")
    assert A.from_dict(d).frames[0].anchorability() == (True, "nut")


def test_legacy_file_without_fields_is_derived():
    d = A.to_dict(_file(_frame(range(0, 4))))
    for fr in d["frames"]:
        del fr["anchorable"], fr["anchor_cue"]
    assert A.from_dict(d).frames[0].anchorability() == (True, "nut")


@pytest.mark.parametrize(
    "fields",
    [
        {"anchorable": True, "anchor_cue": None},
        {"anchorable": False, "anchor_cue": "nut"},
        {"anchorable": True, "anchor_cue": "inlay_1_2_3"},
        {"anchorable": "yes", "anchor_cue": "nut"},
    ],
)
def test_validate_rejects_bad_anchor_fields(fields):
    d = A.to_dict(_file(_frame(range(0, 4))))
    d["frames"][0].update(fields)
    with pytest.raises(ValueError):
        A.from_dict(d)
