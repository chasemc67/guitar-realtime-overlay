"""Copy a FretNet checkpoint with the fret-order gate switched on or off (gro.fretnet.model.order_gate), and/or a
different regression-prior sigma (head queryreg).

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
    ap.add_argument("--wires", action="store_true", help="also the cross-side wire gate (gate='wires')")
    ap.add_argument("--prior-sigma", type=float, default=None, help="head queryreg: sigma (px) of the regression prior")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=True)
    ck["gate"] = False if a.off else "wires" if a.wires else True
    if a.prior_sigma is not None:
        ck["prior_sigma"] = a.prior_sigma
    torch.save(ck, a.dst)
    print(f"wrote {a.dst} (gate={ck['gate']}, prior_sigma={ck.get('prior_sigma')})")


if __name__ == "__main__":
    main()
