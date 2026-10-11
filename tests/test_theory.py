import pytest

from gro.theory import (
    STANDARD_TUNING,
    note_name,
    parse_note,
    pitch_class,
    scale_intervals,
    scale_positions,
    use_flats,
)


def test_parse_note():
    assert parse_note("C") == 0
    assert parse_note("A") == 9
    assert parse_note("C#") == parse_note("Db") == 1
    assert parse_note("Cb") == 11
    assert parse_note("E#") == 5
    with pytest.raises(ValueError):
        parse_note("H")


def test_standard_tuning_open_strings():
    assert [note_name(pitch_class(s, 0)) for s in range(6)] == ["E", "A", "D", "G", "B", "E"]
    assert pitch_class(0, 12) == pitch_class(0, 0)
    assert note_name(pitch_class(0, 5)) == "A"  # 5th fret low E
    assert note_name(pitch_class(4, 1)) == "C"


def test_scale_definitions():
    assert len(scale_intervals("major")) == 7
    assert scale_intervals("minor") == scale_intervals("aeolian") == scale_intervals("natural_minor")
    assert scale_intervals("ionian") == scale_intervals("major")
    assert scale_intervals("minor pentatonic") == (0, 3, 5, 7, 10)
    assert scale_intervals("blues") == (0, 3, 5, 6, 7, 10)
    for mode in ["dorian", "phrygian", "lydian", "mixolydian", "locrian"]:
        assert len(scale_intervals(mode)) == 7
    with pytest.raises(KeyError):
        scale_intervals("nope")


def test_modes_are_rotations_of_major():
    major = scale_intervals("major")
    for i, mode in enumerate(["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]):
        rot = tuple(sorted((x - major[i]) % 12 for x in major))
        assert rot == scale_intervals(mode)


def test_a_minor_pentatonic_positions():
    notes = scale_positions("A", "minor_pentatonic", max_fret=24)
    names = {n.name for n in notes}
    assert names == {"A", "C", "D", "E", "G"}
    pos = {(n.string, n.fret) for n in notes}
    # Box 1 at the 5th fret.
    for p in [(0, 5), (0, 8), (1, 5), (1, 7), (2, 5), (2, 7), (3, 5), (3, 7), (4, 5), (4, 8), (5, 5), (5, 8)]:
        assert p in pos
    assert (0, 6) not in pos
    roots = {(n.string, n.fret) for n in notes if n.is_root}
    assert {(0, 5), (0, 17), (1, 0), (1, 12), (1, 24), (3, 2), (5, 5)} <= roots
    assert all(0 <= n.fret <= 24 for n in notes)
    # Every octave repeats.
    assert all((n.string, n.fret + 12) in pos for n in notes if n.fret <= 12)


def test_counts_and_degrees():
    notes = scale_positions("C", "major", max_fret=11)
    assert len(notes) == 6 * 7  # each string has each scale note once per 12 frets
    assert all(n.degree in (0, 2, 4, 5, 7, 9, 11) for n in notes)
    e = next(n for n in notes if n.name == "E")
    assert e.degree_name == "3"


def test_spelling():
    assert use_flats("F", "major")
    assert not use_flats("G", "major")
    assert use_flats("D", "minor")
    names = {n.name for n in scale_positions("F", "major", max_fret=12)}
    assert "Bb" in names and "A#" not in names
    names = {n.name for n in scale_positions("E", "major", max_fret=12)}
    assert "G#" in names


def test_custom_tuning():
    drop_d = (38,) + STANDARD_TUNING[1:]
    notes = scale_positions("D", "major", tuning=drop_d, max_fret=0)
    assert (0, 0) in {(n.string, n.fret) for n in notes}
