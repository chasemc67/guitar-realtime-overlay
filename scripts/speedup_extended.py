"""Choose a continuation backbone using only VAL, then train one 512 student longer."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/speedup"


def main():
    while not (OUT / "ft_416/selection.json").exists():
        time.sleep(10)
    records = []
    for arch in ["resnet18", "resnet34", "mobilenet_v3_large", "efficientnet_b0"]:
        choice = json.loads((OUT / f"distill_{arch}_640/selection.json").read_text())
        val = choice["val"][str(choice["chosen_threshold"])]
        records.append(dict(arch=arch, val=val))
    selected = max(records, key=lambda r: (r["val"]["numbering"], r["val"]["coverage"], -r["val"]["d13_mean_px"]))
    init = ROOT / f"runs/fretnet/speedup_{selected['arch']}_640/best.pt"
    (OUT / "extended_choice.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                records=records,
                decision="VAL numbering only; no held-out score read",
                imgsz=512,
                additional_epochs=20,
            ),
            indent=2,
        )
    )
    commands = [
        [
            "speedup_train.py",
            "--arch",
            selected["arch"],
            "--imgsz",
            "512",
            "--epochs",
            "20",
            "--name",
            "speedup_extended_512",
            "--init",
            str(init),
        ],
        [
            "speedup_eval.py",
            "--weights",
            "runs/fretnet/speedup_extended_512/best.pt",
            "--imgsz",
            "512",
            "--out",
            "eval/speedup/extended_512",
        ],
        [
            "speedup_eval.py",
            "--weights",
            "runs/fretnet/speedup_extended_512/best.pt",
            "--imgsz",
            "512",
            "--out",
            "eval/speedup/extended_512",
            "--held-only",
        ],
    ]
    for index, command in enumerate(commands):
        with (OUT / f"extended_stage_{index}.log").open("w") as output:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    (OUT / "extended.done").write_text("Completed frozen extended student experiment.\n")


if __name__ == "__main__":
    main()
