"""Hand-placed anchors for sample frames (M1), stored as `fretboard-kpts/v1` JSON in samples/anchors/.

The "clicks" below were read off zoomed, grid-labelled crops (scripts/zoom.py) of the *raw* frames. Each wire is given
as a rough point near its bass end and near its treble end; `gro.snap` refines the wire onto the bright ridge and
the anchor is the intersection with the hand-fitted fretboard edge line. Chase's guitars are left-handed and the
camera is not mirrored, so per PLAN D1 the clips are flagged `mirrored: true` and the stored keypoints are in the
mirrored (canonical, right-handed) frame.

    uv run python scripts/hand_anchors.py [--debug]
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


def build(name: str, spec: dict, debug: bool) -> A.LabelFile:
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
        annotator="agent:claude-m1",
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
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in CLICKS.items():
        if a.names and name not in a.names:
            continue
        lf = build(name, spec, a.debug)
        A.save(lf, OUT / f"{name}.json")
        print("wrote", OUT / f"{name}.json", len(lf.frames[0].anchors), "anchors")


if __name__ == "__main__":
    main()
