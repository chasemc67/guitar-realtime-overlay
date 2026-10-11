"""Hand-placed anchors for sample frames (M1), stored as `fretboard-kpts/v1` JSON in samples/anchors/.

The "clicks" below were read off zoomed, grid-labelled crops (scripts/zoom.py) of the *raw* frames. Each wire is given
as a rough point near its bass end and near its treble end; `gro.snap` refines the wire onto the bright ridge and
the anchor is the intersection with the hand-fitted fretboard edge line. Chase's guitars are left-handed and the
camera is not mirrored, so per PLAN D1 the clips are flagged `mirrored: true` and the stored keypoints are in the
mirrored (canonical, right-handed) frame.

    uv run python scripts/hand_anchors.py [--debug]          # samples/anchors (M1 GT)
    uv run python scripts/hand_anchors.py --extra [--debug]  # samples/anchors_extra (extra eval GT)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gro import anchors as A
from gro.geometry import fit_fretboard
from gro.snap import anchors_from_wires

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "samples" / "frames"
OUT = ROOT / "samples" / "anchors"
OUT_EXTRA = ROOT / "samples" / "anchors_extra"

V, OCC = A.VIS_VISIBLE, A.VIS_OCCLUDED

# Raw-frame pixel coordinates. bass edge = the edge nearest the low E string (top of the board in these shots).
CLICKS: dict[str, dict] = {
    "2026-09-03_20-38-02_10s": {
        "video": "2026-09-03_20-38-02.mp4",
        "t": 10.0,
        "guitar": {
            "type": "electric",
            "frets": 24,
            "inlay": "vine",
            "strings": 6,
            "model": "Ibanez JEM (white)",
        },
        "bass_edge": [(290, 236), (350, 242.7), (785.7, 319.3), (1014, 360.7)],
        "treble_edge": [(285, 297), (337.3, 308.3), (773.3, 396), (1005, 440)],
        "wires": {
            0: ((290, 240), (286, 292), V),
            1: ((350, 245), (338, 305), V),
            2: ((404.3, 255), (392, 316), V),
            3: ((454.3, 264), (442.3, 327), V),
            4: ((502.3, 273), (489.3, 335), V),
            5: ((545.7, 280), (534, 345), V),
            6: ((587.3, 288), (575, 353), V),
            7: ((626.5, 294), (617, 343), V),
            8: ((662.3, 300), (651.7, 357), V),
            9: ((696.7, 306), (684, 375), V),
            10: ((728.3, 312), (715.7, 381), V),
            11: ((757.7, 317), (745.7, 387), V),
            12: ((785.7, 322), (773.3, 393), V),
            13: ((811, 326), (800, 397), V),
            14: ((836, 332), (824, 403), V),
            15: ((858.3, 335), (847.3, 407), V),
            16: ((880, 339), (869.3, 411), V),
            17: ((901, 344), (889.3, 415), V),
            18: ((920.7, 346), (909.3, 419), V),
            19: ((938.3, 350), (926.7, 423), V),
            20: ((956, 353), (944, 427), V),
            21: ((971.7, 356), (960.7, 430), V),
            22: ((986, 359), (976, 433), V),
            23: ((1000.7, 361), (989.3, 435), V),
            24: ((1014, 363), (1005, 437), V),
        },
    },
    # Nut hidden under a fret wrap (x ~ 230-295). Fret 1 = first wire right of the wrap: the extrapolated nut gap
    # (~56 px) puts the nut under the wrap, and the count reaches the 24th fret at the board's end.
    "2026-09-06_09-00-01_10s": {
        "video": "2026-09-06_09-00-01.mp4",
        "t": 10.0,
        "guitar": {
            "type": "electric",
            "frets": 24,
            "inlay": "vine",
            "strings": 6,
            "model": "Ibanez (sunburst)",
        },
        "notes": "nut hidden under fret wrap",
        "bass_edge": [(313.3, 284), (633.3, 351.7)],
        "treble_edge": [(787.5, 452.5), (860, 468.75), (940, 487.5)],
        "wires": {
            1: ((312.5, 298.7), (303.4, 341.7), V),
            2: ((365.9, 312.0), (357.4, 355.0), V),
            3: ((417.7, 320.3), (409.6, 363.3), V),
            4: ((462.9, 327.0), (455.1, 370.0), V),
            5: ((504.7, 333.7), (497.3, 376.7), V),
            6: ((544.8, 342.0), (537.8, 385.0), V),
            7: ((581.4, 348.7), (574.8, 391.7), V),
            8: ((614.9, 353.7), (608.6, 396.7), V),
            9: ((645.0, 363.7), (638.9, 406.7), V),
            10: ((673.4, 369.7), (667.5, 412.7), V),
            11: ((701.7, 374.7), (696.1, 417.7), V),
            12: ((728.5, 378.7), (723.1, 421.7), V),
            13: ((751.9, 383.7), (746.7, 426.7), V),
            14: ((775.3, 388.0), (770.4, 431.0), V),
            15: ((797.0, 392.0), (792.2, 435.0), V),
            16: ((817.0, 396.0), (812.4, 439.0), V),
            17: ((836.1, 400.3), (831.7, 443.3), V),
            18: ((853.8, 403.7), (849.6, 446.7), V),
            19: ((870.6, 407.0), (866.4, 450.0), V),
            20: ((886.2, 410.3), (882.2, 453.3), V),
            21: ((900.6, 413.7), (896.8, 456.7), V),
            22: ((914.0, 417.0), (910.2, 460.0), V),
            23: ((927.3, 419.3), (923.7, 462.3), V),
            24: ((939.0, 421.3), (935.5, 464.3), V),
        },
    },
    # Overexposed frame; nut hidden under a fret wrap (between the locking-nut pads at x ~ 90 and the first wire).
    # Frets past ~14 are washed out and not annotated; fret 5 is mostly under a finger (placed from neighbours).
    "2026-08-30_15-06-39_10s": {
        "video": "2026-08-30_15-06-39.mp4",
        "t": 10.0,
        "guitar": {
            "type": "electric",
            "frets": 24,
            "inlay": "vine",
            "strings": 6,
            "model": "Ibanez (sunburst)",
        },
        "notes": "nut hidden under fret wrap; overexposed; frets > 14 not annotated",
        "bass_edge": [(175, 153.3), (245, 171.7), (303.3, 181.7), (447.5, 215), (696, 267.5)],
        "treble_edge": [(166.7, 233.3), (235, 246.7), (295, 258.3), (441, 292.5), (669, 337.5)],
        "wires": {
            1: ((173.8, 165.3), (167.9, 221.3), V),
            2: ((243.5, 182.9), (236.5, 235.4), V),
            3: ((302.1, 193.2), (296.2, 246.8), V),
            4: ((355.4, 208.2), (350.1, 260.8), V),
            5: ((402.2, 221.2), (397.1, 273.8), OCC),
            6: ((446.5, 226.6), (442.0, 280.9), V),
            7: ((487.6, 236.2), (482.2, 288.8), V),
            8: ((525.1, 244.1), (520.9, 298.4), V),
            9: ((559.2, 250.8), (555.8, 306.8), V),
            10: ((590.1, 257.9), (585.9, 313.1), V),
            11: ((619.2, 264.1), (615.8, 318.4), V),
            12: ((647.0, 269.0), (644.5, 322.5), V),
            13: ((672.0, 273.8), (669.5, 326.2), V),
            14: ((695.5, 278.5), (693.5, 330.0), V),
        },
    },
}


# Extra GT for the auto-annotation eval (scripts/eval_auto.py), written to samples/anchors_extra/. Read the same way
# (grid-labelled zoom crops of the raw frame, snapped); no auto output was overlaid while reading them.
CLICKS_EXTRA: dict[str, dict] = {
    # Nut visible (locking nut); 24 wires up to the board end.
    "2026-09-03_20-38-02_60s": {
        "video": "2026-09-03_20-38-02.mp4",
        "t": 60.0,
        "guitar": {"type": "electric", "frets": 24, "inlay": "vine", "strings": 6, "model": "Ibanez JEM (white)"},
        "notes": "fretting hand over frets 4-7",
        "bass_edge": [
            (338, 257),
            (386, 265),
            (432, 272),
            (591, 299),
            (625, 304),
            (747, 324),
            (821.6, 337),
            (904, 349),
            (985, 362),
        ],
        "treble_edge": [(329.4, 316), (648, 376), (734, 395), (808, 408), (908, 429), (971, 438)],
        "wires": {
            0: ((287, 252), (291, 308), V),
            1: ((338, 258), (329.4, 316), V),
            2: ((386, 266), (376, 326), V),
            3: ((432, 273), (422, 330), V),
            4: ((475, 280), (465, 337), OCC),
            5: ((516, 287), (507, 344), OCC),
            6: ((553, 294), (545, 350), OCC),
            7: ((591, 299), (583, 355), V),
            8: ((625, 305), (616, 364), V),
            9: ((658, 310), (648, 376), V),
            10: ((689, 315), (678, 383), V),
            11: ((718, 320), (706, 389), V),
            12: ((747, 324), (734, 395), V),
            13: ((773, 329), (761, 399), V),
            14: ((798, 332), (786, 405), V),
            15: ((821.6, 337), (808, 408), V),
            16: ((843.6, 340), (830, 414), V),
            17: ((865, 343), (851, 418), V),
            18: ((885, 346), (871, 423), V),
            19: ((904, 349), (890, 425), V),
            20: ((922, 352), (908, 429), V),
            21: ((939, 355), (925, 432), V),
            22: ((955, 357), (941, 434), V),
            23: ((971, 360), (957, 436), V),
            24: ((985, 362), (971, 438), V),
        },
    },
    # Nut under the fret wrap; fret 1 = first wire right of the wrap (spacing extrapolates the nut under it, and the
    # count reaches 24 one gap before the fingerboard end at x ~ 941). Fret 11 is under a finger.
    "2026-09-06_09-00-01_60s": {
        "video": "2026-09-06_09-00-01.mp4",
        "t": 60.0,
        "guitar": {"type": "electric", "frets": 24, "inlay": "vine", "strings": 6, "model": "Ibanez (sunburst)"},
        "notes": "nut hidden under fret wrap; fretting hand over frets 9-13",
        "bass_edge": [
            (291.3, 323.8),
            (348.8, 331.3),
            (402.5, 340),
            (495, 352.5),
            (610, 366.3),
            (725, 381.25),
            (940, 410),
        ],
        "treble_edge": [
            (278.8, 397.5),
            (390, 410),
            (482.5, 423.8),
            (597.5, 438.8),
            (762.5, 462.5),
            (875, 480),
            (912.5, 483.75),
        ],
        "wires": {
            1: ((291.3, 323.8), (278.8, 397.5), V),
            2: ((348.8, 331.3), (337.5, 401.3), V),
            3: ((402.5, 340), (390, 410), V),
            4: ((450, 345), (437.5, 416.3), V),
            5: ((495, 352.5), (482.5, 423.8), V),
            6: ((536.3, 357.5), (523.8, 430), V),
            7: ((575, 361.3), (562.5, 432.5), V),
            8: ((610, 366.3), (597.5, 438.8), V),
            9: ((642.5, 370), (630, 442), OCC),
            10: ((671.3, 373.8), (661, 447.5), V),
            11: ((698, 378), (686, 451), OCC),
            12: ((725, 381.2), (712, 455.4), OCC),
            13: ((749, 384.5), (736, 458.8), V),
            14: ((772.5, 387.6), (759.5, 462.1), V),
            15: ((794, 390.5), (781, 465.1), V),
            16: ((814, 393.1), (801, 467.9), V),
            17: ((832.5, 395.6), (819.5, 470.5), V),
            18: ((850, 398.0), (837, 473.0), V),
            19: ((865, 400.0), (852, 475.1), V),
            20: ((880, 402.0), (867, 477.2), V),
            21: ((894, 403.8), (881, 479.2), V),
            22: ((907.5, 405.7), (894.5, 481.1), V),
            23: ((920, 407.3), (907, 482.9), V),
            24: ((931, 408.8), (918, 484.4), V),
        },
    },
}


def build(name: str, spec: dict, debug: bool, annotator: str = "agent:claude-m1") -> A.LabelFile:
    img = cv2.imread(str(FRAMES / f"{name}.jpg"))
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    raw, info = anchors_from_wires(gray, spec["bass_edge"], spec["treble_edge"], spec["wires"])
    if debug:
        dbg = img.copy()
        for _line, pts, keep in info.pop("edges").values():
            for p, k in zip(pts, keep, strict=True):
                cv2.drawMarker(
                    dbg,
                    tuple(np.int32(np.round(p))),
                    (255, 255, 0) if k else (0, 128, 255),
                    cv2.MARKER_CROSS,
                    6,
                )
        for d in info.values():
            r, s = d["rough"], d["snapped"]
            cv2.line(dbg, tuple(np.int32(r[0])), tuple(np.int32(r[1])), (0, 0, 255), 1)
            cv2.line(dbg, tuple(np.int32(s[0])), tuple(np.int32(s[1])), (0, 255, 0), 1)
        for a in raw:
            cv2.circle(dbg, (int(round(a.x)), int(round(a.y))), 2, (255, 0, 255), -1)
        cv2.imwrite(str(ROOT / "samples" / "out" / f"{name}_clicks.jpg"), dbg)
    canon = A.mirror_anchors(raw, w)  # D1: left-handed clip -> canonical (no bass/treble swap)
    m = fit_fretboard([a for a in canon if a.vis > 0], tau=None)  # recomputed by tooling, never hand-edited
    fit = {"H": np.round(m.H, 8).tolist(), "tau": m.tau, "rmse_px": round(m.rmse_px, 3), "index_shift": 0}
    frame = A.FrameLabel(
        t=spec["t"],
        anchors=canon,
        fit=fit,
        source="manual",
        annotator=annotator,
        status="draft",
        notes="; ".join(
            filter(
                None,
                [
                    "hand-read from zoomed crops, snapped to wire ridges (scripts/hand_anchors.py)",
                    spec.get("notes"),
                ],
            )
        ),
        extra={"frame_file": f"samples/frames/{name}.jpg"},
    )
    clip = A.LabelFile(
        clip_id=name,
        video={"path": f"data/raw/{spec['video']}", "w": w, "h": h, "fps": 60, "mirrored": True},
        guitar=spec["guitar"],
        frames=[frame],
    )
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--extra", action="store_true", help="build the extra eval GT (samples/anchors_extra)")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    clicks, out, who = (
        (CLICKS_EXTRA, OUT_EXTRA, "agent:claude-auto-eval") if a.extra else (CLICKS, OUT, "agent:claude-m1")
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in clicks.items():
        if a.names and name not in a.names:
            continue
        lf = build(name, spec, a.debug, who)
        A.save(lf, out / f"{name}.json")
        print("wrote", out / f"{name}.json", len(lf.frames[0].anchors), "anchors")


if __name__ == "__main__":
    main()
