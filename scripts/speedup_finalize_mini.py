"""Select compression/fast-tracking candidates from VAL and latency only, then execute."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent
CANDIDATES = [
    "distill_resnet18_640",
    "distill_resnet34_640",
    "distill_mobilenet_v3_large_640",
    "distill_efficientnet_b0_640",
    "ft_640",
    "ft_512",
    "ft_416",
]


def main():
    while not (OUT / "video.done").exists():
        time.sleep(10)
    records = []
    for name in CANDIDATES:
        selection = json.loads((OUT / f"selection_{name}.json").read_text())
        # Never inspect the held-out metrics when selecting models.
        timing = json.loads((OUT / f"held_{name}.json").read_text())["latency"]["CPU_AND_NE"]["median_ms"]
        val = selection["val"][str(selection["chosen_threshold"])]
        records.append(dict(name=name, val=val, median_ms=timing))

    def key(row):
        return row["val"]["numbering"], row["val"]["coverage"], -row["val"]["d13_mean_px"]

    best = max(records, key=key)
    fast_enough = [r for r in records if r["median_ms"] <= 10]
    fast = max(fast_enough, key=key) if fast_enough else min(records, key=lambda r: r["median_ms"])
    near_target = [row for row in records if row["median_ms"] <= 8.5]
    near = max(near_target, key=key) if near_target else fast
    chosen = list(dict.fromkeys([best["name"], fast["name"], near["name"]]))
    fast = near  # Track the strongest VAL candidate already near the user's ~8 ms target.
    (OUT / "compressed_candidates.json").write_text(
        json.dumps(
            dict(
                records=records, best=best["name"], fast=fast["name"], chosen=chosen, selection="VAL plus latency only"
            ),
            indent=2,
        )
    )

    def run(script, args, log):
        with (OUT / log).open("w") as output:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for name in chosen:
        for kind in ["int8", "palette6", "palette4", "w8a8"]:
            model = OUT / f"{name}_{kind}.mlpackage"
            run(
                "speedup_compress.py",
                ["--model", OUT / f"{name}.mlpackage", "--kind", kind, "--bundle", OUT / "bundle", "--out", model],
                f"compress_{name}_{kind}.log",
            )
            if model.exists():
                run(
                    "speedup_coreml.py",
                    [
                        "--model",
                        model,
                        "--selection",
                        OUT / f"selection_{name}.json",
                        "--bundle",
                        OUT / "bundle",
                        "--pt",
                        OUT / f"{name}.pt",
                        "--out",
                        OUT / f"held_{name}_{kind}.json",
                    ],
                    f"held_{name}_{kind}.log",
                )
            print(name, kind, "finished", flush=True)
    name = fast["name"]
    run(
        "speedup_tracking.py",
        [
            "--model",
            OUT / f"{name}.mlpackage",
            "--selection",
            OUT / f"selection_{name}.json",
            "--bundle",
            OUT / "bundle",
            "--out",
            OUT / f"tracking_{name}.json",
        ],
        f"tracking_{name}.log",
    )
    clips = Path.home() / "dev/jam-dashboard/clips"
    run(
        "speedup_video.py",
        [
            "--model",
            OUT / f"{name}.mlpackage",
            "--selection",
            OUT / f"selection_{name}.json",
            "--tracking-selection",
            OUT / f"tracking_{name}.selection.json",
            "--clip",
            f"{clips}/2026-09-06_09-00-01.mp4:1:100",
            "--clip",
            f"{clips}/yt_nBpe2YQEzZo.mp4:0:5",
            "--clip",
            f"{clips}/stan_head3s.mp4:1:0",
            "--out",
            OUT / f"video_{name}.json",
        ],
        f"video_{name}.log",
    )
    (OUT / "finished").write_text("All planned Mini experiments finished.\n")


if __name__ == "__main__":
    main()
