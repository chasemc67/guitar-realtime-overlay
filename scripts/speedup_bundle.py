"""Package lossless held-out frames and true predecessors, plus VAL-only calibration."""

# ruff: noqa: E402
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev

from gro import anchors as A


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / "configs/eval_v2.json").read_text())
    rows = []

    def add(image, prev, gt, item, clip, group, index):
        n = len(rows)
        cv2.imwrite(str(a.out / f"{n}.png"), image)
        if prev is not None:
            cv2.imwrite(str(a.out / f"{n}_prev.png"), prev)
        rows.append(
            dict(
                image=f"{n}.png",
                previous=f"{n}_prev.png" if prev is not None else None,
                item=item,
                clip=clip,
                group=group,
                frame_index=index,
                gt=[asdict(g) for g in gt],
            )
        )

    for clip in cfg["clips"]:
        p = ROOT / cfg["gt_dir"] / f"{clip['id']}.json"
        if not p.exists() or cfg.get("checks", {}).get(clip["id"], {}).get("exclude"):
            continue
        lf = A.load(p)
        indices = [int(f.extra["frame_index"]) for f in lf.frames]
        imgs = ev.read_frames(
            ROOT / clip["video"], sorted(set(indices + [i - 1 for i in indices if i > 0])), clip["mirrored"]
        )
        for fr in lf.frames:
            i = int(fr.extra["frame_index"])
            if i in imgs:
                add(imgs[i], imgs.get(i - 1), fr.present(), f"{clip['id']}#{i}", clip["id"], "eval_v2", i)
    for path, lf, i, fr in ev.load_gt([ROOT / g for g in cfg["honest"]]):
        image = ev.frame_image(lf, fr, path)
        if image is None:
            raise RuntimeError(f"Missing honest frame {path}")
        canon, gt = ev.canonical(image, lf, fr.anchors)
        cap = cv2.VideoCapture(str(ROOT / lf.video["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fi = int(fr.extra.get("frame_index", round(fr.t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - 1))
        ok, prev = cap.read()
        cap.release()
        if ok and lf.video.get("mirrored", False):
            prev = cv2.flip(prev, 1)
        add(canon, prev if ok else None, [g for g in gt if g.vis > 0], f"{path.name}#{i}", path.stem, "honest_0906", fi)
    assert len([r for r in rows if r["group"] == "eval_v2"]) == 232
    assert len([r for r in rows if r["group"] == "honest_0906"]) == 15
    (a.out / "manifest.json").write_text(json.dumps(rows, indent=1))
    # Evenly spaced deterministic VAL subset: calibration/parity only, no held-out selection.
    paths = sorted((ROOT / "data/datasets/scale-v1aug/images/val").glob("*.jpg"))
    assert len(paths) == 1498
    forbidden = [c["id"] for c in cfg["clips"]] + ["09-06", "09_06"]
    calibration = []
    for p in paths[::47]:
        assert not any(s in p.name for s in forbidden)
        name = f"calibration_{len(calibration)}.png"
        cv2.imwrite(str(a.out / name), cv2.imread(str(p)))
        calibration.append(dict(image=name, source=p.name))
    (a.out / "calibration.json").write_text(json.dumps(calibration, indent=1))
    print(f"Packaged {len(rows)} held-out frames and {len(calibration)} VAL calibration frames", flush=True)


if __name__ == "__main__":
    main()
