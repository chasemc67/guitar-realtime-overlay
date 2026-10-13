"""Check collected experiment artifacts without evaluating models again."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/speedup"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    manifest_path = OUT / "bundle_manifest.json"
    manifest = read(manifest_path)
    expected_ids = [row["item"] for row in manifest]
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert len(expected_ids) == len(set(expected_ids)) == 247
    assert len(read(OUT / "calibration_manifest.json")) == 32
    models = []
    for path in sorted(OUT.glob("ml_held_*.json")):
        report = read(path)
        assert report["bundle_manifest_sha256"] == manifest_hash, path
        assert report["warmups"] == 20 and report["calls"] == 200, path
        for units in ("CPU_AND_NE", "ALL"):
            held = report["held"][units]
            assert held["macro"]["frames"] == 232 and held["macro"]["clips"] == 14, path
            assert held["honest_0906"]["frames"] == 15, path
            rows = held["frames"]["eval_v2"] + held["frames"]["honest_0906"]
            assert [row["item"] for row in rows] == expected_ids, path
            for variant in ("raw", "refit"):
                assert report["parity"][units][variant]["frames"] == 32, path
            raw = np.load(path.with_suffix(f".{units}.npz"))["raw"]
            assert raw.shape == (247, 155, 1) and np.isfinite(raw).all(), path
            timing = report["latency"][units]
            assert 0 < timing["median_ms"] <= timing["p95_ms"], path
        models.append(path.stem.removeprefix("ml_held_"))
    selections = sorted(OUT.glob("selection_*.json")) + sorted(OUT.glob("resize_*.json"))
    for path in selections:
        selection = read(path)
        best = max(
            selection["val"],
            key=lambda t: (
                selection["val"][t]["numbering"],
                selection["val"][t]["coverage"],
                -selection["val"][t]["d13_mean_px"],
            ),
        )
        assert float(best) == selection["chosen_threshold"], path
        assert all(v["frames"] == 1498 for v in selection["val"].values()), path
    tracks = []
    for path in sorted(OUT.glob("ml_tracking_*.json")):
        if ".selection." in path.name:
            continue
        report = read(path)
        assert report["macro"]["frames"] == 228, path
        assert report["honest_0906"]["frames"] == 15, path
        assert len(report["excluded_no_predecessor"]) == 4, path
        tracks.append(path.stem.removeprefix("ml_tracking_"))
    videos = []
    for path in sorted(OUT.glob("ml_video_*.json")):
        report = read(path)
        assert len(report["clips"]) == 3, path
        assert [c["frames"] for c in report["clips"]] == [120, 120, 90], path
        videos.append(path.stem.removeprefix("ml_video_"))
    if args.complete:
        assert len(models) == 36, models
        assert len(selections) == 12, len(selections)
        assert len(list(OUT.glob("held_*.json"))) == 12
        assert len(tracks) == len(videos) == len(list(OUT.glob("ml_control_*.json"))) == 7
    result = dict(
        complete=args.complete,
        coreml_variants=len(models),
        compute_modes_per_variant=2,
        held_frames_per_mode=247,
        parity_frames_per_mode=32,
        val_selections=len(selections),
        tracking_variants=len(tracks),
        contiguous_video_variants=len(videos),
        models=models,
        checks="Frame identities/counts, finite raw outputs, parity coverage, timing, VAL argmax, manifest hash",
    )
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
