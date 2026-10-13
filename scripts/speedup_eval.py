"""Size-specific VAL selection and one frozen held-out evaluation; no app changes."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev
from val_refit import score

from gro.anchors import Anchor
from gro.fretnet.data import read_label
from gro.fretnet.numbering import anchors_from_k, reassign_wires
from gro.geometry import FitError, fit_fretboard
from gro.infer import Letterbox, decode, load_backend
from gro.infer.backends import Backend


def refit(raw, threshold):
    raw = raw.copy()
    k, conf, _ = decode(raw)
    if conf >= 0.25 and np.isfinite(k).all():
        try:
            fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
            raw[5:, 0] = reassign_wires(k, fit, tolerance=threshold, reject_ambiguous=True).reshape(-1)
        except (FitError, ValueError, np.linalg.LinAlgError):
            pass
    return raw


class RefitBackend(Backend):
    def __init__(self, backend, threshold):
        self.backend, self.threshold, self.imgsz = backend, threshold, backend.imgsz

    def __call__(self, rgb):
        return refit(self.backend(rgb), self.threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/datasets/scale-v1aug")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--held-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    backend = load_backend(a.weights, device=a.device, imgsz=a.imgsz)
    sha = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    if not a.held_only:
        if (a.out / "selection.json").exists():
            ap.error("Selection already frozen; use --held-only")
        paths = sorted((a.data / "images/val").glob("*.jpg"))
        assert len(paths) == 1498
        cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
        forbidden = [c["id"] for c in cfg["clips"]] + ["09-06", "09_06"]
        assert not any(s in p.name for p in paths for s in forbidden)
        protocol = dict(
            weights_sha256=sha,
            imgsz=a.imgsz,
            thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            val_manifest_sha256=hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest(),
            selection="VAL numbering, coverage, negative D13 mean",
            seed=0,
            ransac_iters=100,
        )
        (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
        rows = {t: [] for t in protocol["thresholds"]}
        failed = 0
        for idx, p in enumerate(paths):
            image = cv2.imread(str(p))
            h, w = image.shape[:2]
            g = read_label(a.data / "labels/val" / (p.stem + ".txt"), w, h)
            gt = [
                Anchor(i // 2, ("bass", "treble")[i % 2], float(x), float(y), vis=int(v))
                for i, (x, y, v) in enumerate(g)
                if v > 0
            ]
            lb = Letterbox.for_shape(image.shape, a.imgsz)
            k, conf, _ = decode(backend(cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB)), 0.25)
            detected = bool(conf >= 0.25 and np.isfinite(k).all())
            fit = None
            if detected:
                try:
                    fit = fit_fretboard(anchors_from_k(k), ransac_iters=100)
                except (FitError, ValueError, np.linalg.LinAlgError):
                    failed += 1
            for t in rows:
                out = reassign_wires(k, fit, tolerance=t, reject_ambiguous=True) if fit else k.copy()
                out[:, :2] = lb.to_frame(out[:, :2])
                rows[t].append(score(gt, out if detected else np.zeros_like(k), detected, p.name))
            if idx % 100 == 0:
                print(f"VAL {a.imgsz} {idx}/{len(paths)}", flush=True)
        summary = {str(t): ev.agg(rs) for t, rs in rows.items()}
        chosen = max(
            rows,
            key=lambda t: (summary[str(t)]["numbering"], summary[str(t)]["coverage"], -summary[str(t)]["d13_mean_px"]),
        )
        selection = dict(**protocol, chosen_threshold=chosen, fit_failures=failed, val=summary)
        (a.out / "selection.json").write_text(json.dumps(selection, indent=2))
        print(json.dumps(selection, indent=2), flush=True)
    else:
        if (a.out / "held.json").exists() or (a.out / "held.started").exists():
            ap.error("Held-out evaluation already attempted; never silently rerun")
        selection = json.loads((a.out / "selection.json").read_text())
        assert selection["weights_sha256"] == sha and selection["imgsz"] == a.imgsz
        (a.out / "held.started").write_text("Frozen selection loaded; held-out evaluation started.\n")
        locked = RefitBackend(backend, selection["chosen_threshold"])
        ev.load_backend = lambda *args, **kwargs: locked
        sys.argv = ["eval_v2.py", "--weights", str(a.weights), "--name", "held", "--out", str(a.out)]
        ev.main()


if __name__ == "__main__":
    main()
