import csv

import cv2
import numpy as np

from gro import anchors as A
from gro.geometry import dlt, fret_u, model_from_h, plane_point
from gro.review import run_review


def _model():
    src = plane_point(np.array([0, 0, 0.7, 0.7]), np.array([0, 1, 0, 1]), tau=0.3)
    dst = np.array([[600, 100], [620, 160], [40, 260], [70, 380]], float)
    return model_from_h(dlt(src, dst))


def _label(tmp_path, clip, mirrored, outlier):
    m = _model()
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (400, 640, 3), dtype=np.uint8)
    anchors = []
    for n in range(2, 13):
        for side, y in (("bass", 0.0), ("treble", 1.0)):
            p = m.project(fret_u(n), y)
            vis = A.VIS_OCCLUDED if n == 5 else A.VIS_VISIBLE
            anchors.append(A.Anchor(n, side, float(p[0]), float(p[1]), vis))
    if outlier:
        a = anchors[10]
        anchors[10] = A.Anchor(a.fret, a.side, a.x + 25, a.y - 15, a.vis)
    # Store the original-orientation frame; mirrored clips keep canonical (mirrored) coordinates.
    cv2.imwrite(str(tmp_path / f"{clip}.png"), cv2.flip(frame, 1) if mirrored else frame)
    lf = A.LabelFile(
        clip,
        {"path": "missing.mp4", "w": 640, "h": 400, "fps": 60, "mirrored": mirrored},
        {"type": "electric", "frets": 24},
        [A.FrameLabel(t=1.0, anchors=anchors, extra={"frame_file": f"{clip}.png"})],
    )
    A.save(lf, tmp_path / f"{clip}.json")


def test_review_batch(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    _label(labels, "clean", mirrored=False, outlier=False)
    _label(labels, "lefty", mirrored=True, outlier=True)
    res = run_review(labels, tmp_path / "review", "batch-test", bases=[tmp_path], cols=2, rows=1, width=1200)

    out = tmp_path / "review" / "batch-test"
    assert res["out"] == out
    for name in ("clean_f000", "lefty_f000"):
        assert (out / "frames" / f"{name}.jpg").exists()
        strip = cv2.imread(str(out / "strips" / f"{name}.jpg"))
        assert strip is not None and strip.shape[1] == 1200
    sheet = cv2.imread(str(out / "contact_sheet_01.jpg"))
    assert sheet.shape[1] == 1200
    assert not (out / "contact_sheet_02.jpg").exists()

    stats = {s["item"]: s for s in res["items"]}
    clean, lefty = stats["clean_f000"], stats["lefty_f000"]
    assert clean["max_px"] < 0.1 and clean["n_rejected"] == 0 and clean["nut_visible"] == "no"
    assert lefty["n_rejected"] == 1 and lefty["rejected"] == "7b" and lefty["mean_px"] < 0.1
    rows = list(csv.DictReader((out / "summary.csv").open()))
    assert [r["item"] for r in rows] == ["clean_f000", "lefty_f000"]

    # Full-res review image = caption bar + the frame at native resolution.
    a = cv2.imread(str(out / "frames" / "clean_f000.jpg"))
    assert a.shape[1] == 640 and a.shape[0] > 400


def test_review_two_sheets(tmp_path):
    for i in range(3):
        _label(tmp_path, f"c{i}", mirrored=False, outlier=False)
    run_review(tmp_path, tmp_path / "review", "b", bases=[tmp_path], cols=1, rows=2, width=800)
    assert (tmp_path / "review" / "b" / "contact_sheet_02.jpg").exists()
