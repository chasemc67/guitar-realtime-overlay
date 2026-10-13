"""Choose a continuation backbone using only VAL, then train a 512/640 student longer."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/speedup"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=[512, 640], default=512)
    size = parser.parse_args().size
    name = f"extended_{size}"
    choice_name = "extended_choice" if size == 512 else name + "_choice"
    done_name = "extended.done" if size == 512 else name + ".done"
    if size == 640:
        while not (OUT / "extended.done").exists():
            time.sleep(10)
    while not (OUT / "ft_416/selection.json").exists():
        time.sleep(10)
    records = []
    for arch in ["resnet18", "resnet34", "mobilenet_v3_large", "efficientnet_b0"]:
        choice = json.loads((OUT / f"distill_{arch}_640/selection.json").read_text())
        val = choice["val"][str(choice["chosen_threshold"])]
        records.append(dict(arch=arch, val=val))
    selected = max(records, key=lambda r: (r["val"]["numbering"], r["val"]["coverage"], -r["val"]["d13_mean_px"]))
    init = ROOT / f"runs/fretnet/speedup_{selected['arch']}_640/best.pt"
    (OUT / f"{choice_name}.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                records=records,
                decision="VAL numbering only; no held-out score read",
                imgsz=size,
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
            str(size),
            "--epochs",
            "20",
            "--name",
            f"speedup_{name}",
            "--init",
            str(init),
        ],
        [
            "speedup_eval.py",
            "--weights",
            f"runs/fretnet/speedup_{name}/best.pt",
            "--imgsz",
            str(size),
            "--out",
            f"eval/speedup/{name}",
        ],
        [
            "speedup_eval.py",
            "--weights",
            f"runs/fretnet/speedup_{name}/best.pt",
            "--imgsz",
            str(size),
            "--out",
            f"eval/speedup/{name}",
            "--held-only",
        ],
    ]
    for index, command in enumerate(commands):
        with (OUT / f"{name}_stage_{index}.log").open("w") as output:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    (OUT / done_name).write_text("Completed frozen extended student experiment.\n")


if __name__ == "__main__":
    main()
