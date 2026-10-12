"""Copy a FretNet checkpoint with the fret-order gate switched on or off (gro.fretnet.model.order_gate).

    .venv/bin/python scripts/gate_checkpoint.py runs/fretnet/v3b_r50/best.pt runs/fretnet/v3b_r50/best_gate.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--off", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=True)
    ck["gate"] = not a.off
    torch.save(ck, a.dst)
    print(f"wrote {a.dst} (gate={ck['gate']})")


if __name__ == "__main__":
    main()
