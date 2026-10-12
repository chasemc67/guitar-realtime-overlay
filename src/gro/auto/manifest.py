"""Dataset manifests (`data_manifests/*.tsv`): video ids flagged in the `exclude` column (e.g. "acoustic"; the target is
electric guitars) are skipped by the auto-annotation pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[3] / "data_manifests"


def excluded_ids(manifest_dir: Path = MANIFEST_DIR) -> dict[str, str]:
    """video id -> exclusion reason, over every manifest."""
    out: dict[str, str] = {}
    for p in sorted(Path(manifest_dir).glob("*.tsv")):
        with p.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if (row.get("exclude") or "").strip():
                    out[row["id"]] = row["exclude"].strip()
    return out


def video_id(clip_id: str) -> str:
    """`<videoid>_<t>` (scraped stills) -> `<videoid>`."""
    parts = clip_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].replace(".", "").isdigit() else clip_id


def _flag(value: str | None) -> bool | None:
    v = (value or "").strip().lower()
    return True if v in ("yes", "true", "1") else False if v in ("no", "false", "0") else None


def clip_info(video: str | Path, manifest_dir: Path = MANIFEST_DIR) -> dict:
    """Per-clip flags for a video path: `mirrored` (D1; bool, default False) and `frets` (known fret count or None =
    detect). Matches the manifest `id` against the file stem, with or without the `yt_` prefix."""
    stem = Path(video).stem
    keys = {stem, stem.removeprefix("yt_"), video_id(stem)}
    for p in sorted(Path(manifest_dir).glob("*.tsv")):
        with p.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("id") in keys:
                    frets = (row.get("frets") or "").strip()
                    return {
                        "id": row["id"],
                        "manifest": p.name,
                        "mirrored": bool(_flag(row.get("mirrored"))),
                        "frets": int(frets) if frets.isdigit() else None,
                    }
    return {"id": stem, "manifest": None, "mirrored": False, "frets": None}
