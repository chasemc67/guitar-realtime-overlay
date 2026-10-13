"""Save Python refit reference fixtures and NumPy seed-0 RANSAC samples for Swift."""

# ruff: noqa: E402
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gro.fretnet.numbering import anchors_from_k, reassign_wires
from gro.geometry import fit_fretboard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    paths = sorted(a.cache.glob("*.npz"))
    fixtures = []
    for i in np.linspace(0, len(paths) - 1, 50).astype(int):
        k = np.load(paths[i])["k"]
        fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
        fixtures.append(
            dict(
                item=paths[i].stem,
                k=k.ravel().tolist(),
                H=fit.H.ravel().tolist(),
                expected=reassign_wires(k, fit, tolerance=0.3, reject_ambiguous=True).ravel().tolist(),
            )
        )
    samples = []
    for n in range(5, 51):
        rng = np.random.default_rng(0)
        samples.extend(
            np.array([rng.choice(np.arange(n), 4, replace=False) for _ in range(100)], dtype=np.uint8).ravel().tolist()
        )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "refit_reference.json").write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
    (a.out / "samples.b64").write_text(base64.b64encode(bytes(samples)).decode())


if __name__ == "__main__":
    main()
