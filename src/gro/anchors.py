"""Anchor (keypoint) label schema `fretboard-kpts/v1` (PLAN §3.1).

A label file holds one clip; each frame has `kpts`: slot name -> {"bass": [x, y, vis], "treble": [x, y, vis]}.
Slots are `nut` (= fret 0) and `f1`..`f24`. Coordinates are full-frame pixels in the *canonical* (right-handed)
orientation: if `video.mirrored` is true, they refer to the horizontally mirrored frame (D1).
vis: 2 = visible, 1 = occluded but inferable, 0 = absent / out of frame.

Each frame also carries `anchorable` / `anchor_cue` (PLAN D12): absolute fret numbering needs a visible cue, checked
in order: the nut, the 9-12-15 inlay frets, or all of the 3-5-7 inlay frets. A cue counts only if every fret it
names has at least one anchor with vis 2. If a file leaves the fields unset, they are derived from the anchors;
explicit values that differ from the derived ones are kept as an override.
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
# D12 anchor cues in priority order: cue name -> frets whose anchors must all be visible.
ANCHOR_CUES: dict[str, tuple[int, ...]] = {"nut": (0,), "inlay_9_12_15": (9, 12, 15), "inlay_3_5_7": (3, 5, 7)}


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
    anchorable: bool | None = None  # None = derive from the anchors (D12)
    anchor_cue: str | None = None
    fit: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def present(self) -> list[Anchor]:
        return [a for a in self.anchors if a.vis > 0]

    def derived_anchor_cue(self) -> str | None:
        """First D12 cue whose frets all have a visible (vis 2) anchor, or None."""
        if not self.neck_present:
            return None
        seen = {a.fret for a in self.anchors if a.vis == VIS_VISIBLE}
        return next((cue for cue, frets in ANCHOR_CUES.items() if seen.issuperset(frets)), None)

    def anchorability(self) -> tuple[bool, str | None]:
        """(anchorable, anchor_cue): the explicit values if set, else derived from the anchors."""
        if self.anchorable is None:
            cue = self.derived_anchor_cue()
            return cue is not None, cue
        return self.anchorable, self.anchor_cue

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
        anchorable, cue = fr.get("anchorable"), fr.get("anchor_cue")
        if anchorable is not None and not isinstance(anchorable, bool):
            raise ValueError(f"anchorable must be a bool, got {anchorable!r}")
        if cue is not None and cue not in ANCHOR_CUES:
            raise ValueError(f"bad anchor_cue {cue!r}")
        if anchorable is not None and anchorable != (cue is not None):
            raise ValueError(f"anchorable={anchorable} inconsistent with anchor_cue={cue!r}")


_FRAME_KEYS = {
    "t",
    "kpts",
    "source",
    "annotator",
    "status",
    "neck_present",
    "anchorable",
    "anchor_cue",
    "fit",
    "flags",
    "notes",
}


def _frame_from_dict(fr: dict[str, Any]) -> FrameLabel:
    label = FrameLabel(
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
    if fr.get("anchorable") is not None or fr.get("anchor_cue") is not None:
        cue = fr.get("anchor_cue")
        explicit = (cue is not None, cue)
        # Stored values that match the anchors stay derived, so they follow later kpt edits.
        if explicit != label.anchorability():
            label.anchorable, label.anchor_cue = explicit
    return label


def from_dict(d: dict[str, Any]) -> LabelFile:
    validate(d)
    frames = [_frame_from_dict(fr) for fr in d["frames"]]
    return LabelFile(d["clip_id"], d["video"], d.get("guitar", {}), frames)


def to_dict(lf: LabelFile) -> dict[str, Any]:
    frames = []
    for fr in lf.frames:
        anchorable, cue = fr.anchorability()
        frames.append(
            {
                "t": fr.t,
                "source": fr.source,
                "annotator": fr.annotator,
                "status": fr.status,
                "qa_reviewer": None,
                "neck_present": fr.neck_present,
                "anchorable": anchorable,
                "anchor_cue": cue,
                "kpts": fr.kpts_dict(),
                "fit": fr.fit,
                "flags": fr.flags,
                "notes": fr.notes,
                **fr.extra,
            }
        )
    return {"schema": SCHEMA, "clip_id": lf.clip_id, "video": lf.video, "guitar": lf.guitar, "frames": frames}


def load(path: str | Path) -> LabelFile:
    d = json.loads(Path(path).read_text())
    ok = []
    for fr in d.get("frames", []):
        try:
            for name in fr.get("kpts", {}):
                slot_fret(name)
            ok.append(fr)
        except ValueError:  # e.g. f25 from a mis-numbered relative run: drop the frame
            pass
    d["frames"] = ok
    return from_dict(d)


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
