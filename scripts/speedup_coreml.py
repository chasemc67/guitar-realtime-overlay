"""CoreML Mini timing, compressed-output held-out scoring, and VAL parity."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev
from coreml_parity import compare, summarize
from speedup_eval import RefitBackend, refit

from gro.anchors import Anchor
from gro.infer import Letterbox, decode, load_backend


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--pt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timing-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if a.out.exists() or a.out.with_suffix(".started").exists():
        ap.error("Already evaluated or attempted; refuse overwrite")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".started").write_text("Evaluation started\n")
    package_hash = hashlib.sha256()
    for path in sorted(a.model.rglob("*")):
        if path.is_file():
            package_hash.update(str(path.relative_to(a.model)).encode())
            package_hash.update(path.read_bytes())
    report = dict(
        model=str(a.model),
        package_sha256=package_hash.hexdigest(),
        latency={},
        held={},
        parity={},
        warmups=20,
        calls=200,
    )
    selection = json.loads(a.selection.read_text()) if a.selection else None
    if not a.timing_only:
        assert selection is not None and a.bundle is not None
        report["selection"] = selection
        report["bundle_manifest_sha256"] = hashlib.sha256((a.bundle / "manifest.json").read_bytes()).hexdigest()
    for units in ("CPU_AND_NE", "ALL"):
        backend = load_backend(a.model, compute_units=units)
        ms = backend.latency_ms(200, 20)
        report["latency"][units] = dict(median_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)))
        print(units, report["latency"][units], flush=True)
        if not a.timing_only:
            assert backend.imgsz == selection["imgsz"]
            threshold = selection["chosen_threshold"]
            if a.pt:
                pt = load_backend(a.pt, device="cpu", imgsz=backend.imgsz)
                parity = {"raw": [], "refit": []}
                for row in json.loads((a.bundle / "calibration.json").read_text()):
                    image = cv2.imread(str(a.bundle / row["image"]))
                    lb = Letterbox.for_shape(image.shape, backend.imgsz)
                    rgb = cv2.cvtColor(lb.apply(image), cv2.COLOR_BGR2RGB)
                    p, m = pt(rgb), backend(rgb)
                    for name, pp, mm in [("raw", p, m), ("refit", refit(p, threshold), refit(m, threshold))]:
                        pk, mk = decode(pp)[0], decode(mm)[0]
                        pk[:, :2] = lb.to_frame(pk[:, :2])
                        mk[:, :2] = lb.to_frame(mk[:, :2])
                        parity[name].append(compare(mk, pk, 24))
                report["parity"][units] = {k: summarize(v) for k, v in parity.items()}
                del pt
            a.out.write_text(json.dumps(report, indent=1))
            model = RefitBackend(backend, threshold)
            rows = defaultdict(list)
            raw_predictions = []
            for row in json.loads((a.bundle / "manifest.json").read_text()):
                image = cv2.imread(str(a.bundle / row["image"]))
                gt = [Anchor(**g) for g in row["gt"]]
                rows[row["group"]].append(
                    ev.frame_row(model, image, gt, SimpleNamespace(stage1=None), row["item"], row["clip"])
                )
                raw_predictions.append(model.last_raw.copy())
            np.savez_compressed(a.out.with_suffix(f".{units}.npz"), raw=np.stack(raw_predictions))
            perclip = defaultdict(list)
            for row in rows["eval_v2"]:
                perclip[row["clip"]].append(row)
            report["held"][units] = dict(
                macro=ev.macro({c: ev.agg(rs) for c, rs in perclip.items()}),
                honest_0906=ev.agg(rows["honest_0906"]),
                frames=rows,
            )
        a.out.write_text(json.dumps(report, indent=1))
        del backend


if __name__ == "__main__":
    main()
