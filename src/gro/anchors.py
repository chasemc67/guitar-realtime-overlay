"""Anchor (keypoint) label schema `fretboard-kpts/v1` (PLAN §3.1).

A label file holds one clip; each frame has `kpts`: slot name -> {"bass": [x, y, vis], "treble": [x, y, vis]}.
Slots are `nut` (= fret 0) and `f1`..`f24`. Coordinates are full-frame pixels in the *canonical* (right-handed)
orientation: if `video.mirrored` is true, they refer to the horizontally mirrored frame (D1).
vis: 2 = visible, 1 = occluded but inferable, 0 = absent / out of frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "fretboard-kpts/v1"
SIDES = ("bass", "treble")
MAX_FRETS = 24
VIS_ABSENT, VIS_OCCLUDED, VIS_VISIBLE = 0, 1, 2


def slot_name(fret: int) -> str:
    return "nut" if fret == 0 else f"f{fret}"


def slot_fret(name: str) -> int:
    if name == "nut":
        return 0
    if name.startswith("f") and name[1:].isdigit():
        n = int(name[1:])
        if 1 <= n <= MAX_FRETS:
            return n
    raise ValueError(f"bad slot name {name!r}")


@dataclass(frozen=True)
class Anchor:
    """One keypoint: where fret wire `fret` (0 = nut) meets the `side` edge of the fretboard."""

    fret: int
    side: str  # "bass" | "treble"
    x: float
    y: float
    vis: int = VIS_VISIBLE
    conf: float = 1.0

    @property
    def y_canon(self) -> float:
        """Across-neck canonical coordinate: 0 = bass edge, 1 = treble edge."""
        return 0.0 if self.side == "bass" else 1.0


@dataclass
class FrameLabel:
    t: float | None
    anchors: list[Anchor]
    source: str = "manual"
    annotator: str = ""
    status: str = "draft"
    neck_present: bool = True
    fit: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def present(self) -> list[Anchor]:
        return [a for a in self.anchors if a.vis > 0]

    def kpts_dict(self) -> dict[str, dict[str, list]]:
        out: dict[str, dict[str, list]] = {}
        for a in sorted(self.anchors, key=lambda a: (a.fret, a.side)):
            slot = out.setdefault(slot_name(a.fret), {s: [None, None, 0] for s in SIDES})
            slot[a.side] = [round(a.x, 2), round(a.y, 2), a.vis]
        return out


@dataclass
class LabelFile:
    clip_id: str
    video: dict[str, Any]
    guitar: dict[str, Any]
    frames: list[FrameLabel]

    @property
    def mirrored(self) -> bool:
        return bool(self.video.get("mirrored", False))


def parse_kpts(kpts: dict[str, dict[str, list]]) -> list[Anchor]:
    out = []
    for name, sides in kpts.items():
        fret = slot_fret(name)
        for side in SIDES:
            trip = sides.get(side)
            if trip is None:
                continue
            x, y, vis = trip
            if vis and x is not None and y is not None:
                out.append(Anchor(fret, side, float(x), float(y), int(vis)))
    return out


def validate(d: dict[str, Any]) -> None:
    if d.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {d.get('schema')!r}")
    for key in ("clip_id", "video", "frames"):
        if key not in d:
            raise ValueError(f"missing {key!r}")
    for fr in d["frames"]:
        for name, sides in fr.get("kpts", {}).items():
            slot_fret(name)
            for side, trip in sides.items():
                if side not in SIDES:
                    raise ValueError(f"bad side {side!r} in slot {name}")
                if len(trip) != 3 or trip[2] not in (0, 1, 2):
                    raise ValueError(f"bad triplet {trip!r} in {name}.{side}")
                if trip[2] > 0 and (trip[0] is None or trip[1] is None):
                    raise ValueError(f"visible point without coordinates in {name}.{side}")


_FRAME_KEYS = {"t", "kpts", "source", "annotator", "status", "neck_present", "fit", "flags", "notes"}


def from_dict(d: dict[str, Any]) -> LabelFile:
    validate(d)
    frames = [
        FrameLabel(
            t=fr.get("t"),
            anchors=parse_kpts(fr.get("kpts", {})),
            source=fr.get("source", "manual"),
            annotator=fr.get("annotator", ""),
            status=fr.get("status", "draft"),
            neck_present=fr.get("neck_present", True),
            fit=fr.get("fit"),
            flags=list(fr.get("flags", [])),
            notes=fr.get("notes", ""),
            extra={k: v for k, v in fr.items() if k not in _FRAME_KEYS},
        )
        for fr in d["frames"]
    ]
    return LabelFile(d["clip_id"], d["video"], d.get("guitar", {}), frames)


def to_dict(lf: LabelFile) -> dict[str, Any]:
    frames = []
    for fr in lf.frames:
        frames.append(
            {
                "t": fr.t,
                "source": fr.source,
                "annotator": fr.annotator,
                "status": fr.status,
                "qa_reviewer": None,
                "neck_present": fr.neck_present,
                "kpts": fr.kpts_dict(),
                "fit": fr.fit,
                "flags": fr.flags,
                "notes": fr.notes,
                **fr.extra,
            }
        )
    return {"schema": SCHEMA, "clip_id": lf.clip_id, "video": lf.video, "guitar": lf.guitar, "frames": frames}


def load(path: str | Path) -> LabelFile:
    return from_dict(json.loads(Path(path).read_text()))


def save(lf: LabelFile, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(lf), indent=2) + "\n")


def mirror_anchors(anchors: list[Anchor], width: int, swap_sides: bool = False) -> list[Anchor]:
    """Mirror x -> width-1-x.

    `swap_sides=False` is the D1 ingest/toggle mirror (physical identity is preserved). `swap_sides=True` is the h-flip
    *augmentation*, which also swaps bass<->treble to keep the canonical orientation (D1, §3.4).
    """
    flip = {"bass": "treble", "treble": "bass"}
    return [
        Anchor(a.fret, flip[a.side] if swap_sides else a.side, width - 1 - a.x, a.y, a.vis, a.conf) for a in anchors
    ]
