"""Music theory: notes, tunings, scales, and mapping a key+scale onto the fretboard.

Strings are indexed by physical identity, 0 = low E (bass, string 6) ... 5 = high E (treble, string 1),
matching PLAN §7 (`tuning` lists MIDI notes for strings 6→1).
"""

from __future__ import annotations

from dataclasses import dataclass

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# MIDI note numbers, low (bass) string first.
TUNINGS: dict[str, tuple[int, ...]] = {
    "standard": (40, 45, 50, 55, 59, 64),  # E2 A2 D3 G3 B3 E4
    "eb": (39, 44, 49, 54, 58, 63),
    "drop_d": (38, 45, 50, 55, 59, 64),
    "d_standard": (38, 43, 48, 53, 57, 62),
    "dadgad": (38, 45, 50, 55, 57, 62),
}
STANDARD_TUNING = TUNINGS["standard"]

# Semitone intervals from the root.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    # Modes of the major scale.
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}
SCALE_ALIASES = {
    "minor": "natural_minor",
    "maj": "major",
    "min": "natural_minor",
    "pentatonic": "minor_pentatonic",
}

DEGREE_NAMES = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]

# Keys conventionally spelled with flats.
_FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb"}
_FLAT_MINOR_KEYS = {"D", "G", "C", "F", "Bb", "Eb"}


def parse_note(name: str) -> int:
    """Pitch class (0..11) of a note name like 'A', 'C#', 'Bb', 'E##'."""
    name = name.strip()
    if not name or name[0].upper() not in _LETTER_PC:
        raise ValueError(f"bad note name: {name!r}")
    pc = _LETTER_PC[name[0].upper()]
    for ch in name[1:]:
        if ch in "#♯":
            pc += 1
        elif ch in "b♭":
            pc -= 1
        else:
            raise ValueError(f"bad note name: {name!r}")
    return pc % 12


def scale_intervals(scale: str) -> tuple[int, ...]:
    key = scale.lower().replace(" ", "_").replace("-", "_")
    key = SCALE_ALIASES.get(key, key)
    if key not in SCALES:
        raise KeyError(f"unknown scale {scale!r}; known: {sorted(SCALES)}")
    return SCALES[key]


def use_flats(root: str, scale: str) -> bool:
    """Spell with flats if the root is written with a flat, or the key signature uses flats."""
    if "b" in root[1:]:
        return True
    if "#" in root:
        return False
    ivs = scale_intervals(scale)
    letter = root.strip()[0].upper()
    is_minorish = 3 in ivs and 4 not in ivs
    return letter in (_FLAT_MINOR_KEYS if is_minorish else _FLAT_KEYS)


def note_name(pc: int, flats: bool = False) -> str:
    return (FLAT_NAMES if flats else SHARP_NAMES)[pc % 12]


def pitch_class(string: int, fret: int, tuning: tuple[int, ...] = STANDARD_TUNING) -> int:
    return (tuning[string] + fret) % 12


@dataclass(frozen=True)
class ScaleNote:
    string: int  # 0 = low E (bass) .. 5 = high E (treble)
    fret: int  # 0 = open string
    pc: int
    degree: int  # semitones above the root, 0..11
    name: str

    @property
    def is_root(self) -> bool:
        return self.degree == 0

    @property
    def degree_name(self) -> str:
        return DEGREE_NAMES[self.degree]


def scale_positions(
    root: str,
    scale: str,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = 24,
    min_fret: int = 0,
) -> list[ScaleNote]:
    """All (string, fret) positions of `root` `scale` for frets min_fret..max_fret, sorted by (string, fret)."""
    root_pc = parse_note(root)
    ivs = set(scale_intervals(scale))
    flats = use_flats(root, scale)
    out = []
    for s, open_midi in enumerate(tuning):
        for f in range(min_fret, max_fret + 1):
            pc = (open_midi + f) % 12
            deg = (pc - root_pc) % 12
            if deg in ivs:
                out.append(ScaleNote(s, f, pc, deg, note_name(pc, flats)))
    return out
