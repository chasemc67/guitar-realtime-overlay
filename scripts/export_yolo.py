"""Export the filtered scale labels as a YOLO-pose dataset (brief train_v1, task 3; PLAN §2.2 baseline, §6 splits).

    .venv/bin/python scripts/export_yolo.py                       # data/labels/scale-v1 -> data/datasets/scale-v1
    .venv/bin/python scripts/export_yolo.py --max-per-split 20 --out data/datasets/tiny   # smoke test

- One class `fretboard`; 50 keypoints = nut, f1..f24 x {bass (low-E), treble (high-E)} in that order
  (`0b 0t 1b 1t ... 24b 24t`); YOLO visibility 2 = visible, 1 = occluded (still supervised), 0 = not labelled.
  The box is the keypoints' extent, padded by half a string spacing across and 2% of the image along the neck.
- Canonical right-handed orientation (D1): `mirrored` clips are flipped before export, labels are already canonical.
  The model is right-handed only, so the training config disables fliplr (`flip_idx` is still written, D1's
  bass<->treble swap, so an accidental fliplr > 0 stays geometrically consistent).
- Only frames kept by scripts/filter_scale.py, and only anchorable ones (D12, absolute numbering; the keypoints are
  absolute fret slots); relative-only frames are counted and skipped.
- Split BY VIDEO (configs/scale-v1_splits.json, written on the first run and reused, so the PC and the box agree):
  test = Chase's 2026-09-06 (held out, never exported to train/val) + --n-test held-out YouTube clips; val = ~15% of
  the remaining YouTube clips + Chase's 2026-09-03; train = the rest + Chase's 2026-08-30.
- Images are JPEG q95, longest side <= --max-side (labels are normalised, so resizing is free).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gro import anchors as A  # noqa: E402

N_FRETS = 24
N_KPTS = 2 * (N_FRETS + 1)
KPT_NAMES = [f"{n}{s}" for n in range(N_FRETS + 1) for s in ("b", "t")]
FLIP_IDX = [i ^ 1 for i in range(N_KPTS)]  # D1: h-flip swaps bass <-> treble for every slot
SPLITS = ROOT / "configs" / "scale-v1_splits.json"
TEST_OWNER = "2026-09-06_09-00-01"
VAL_OWNER = "2026-09-03_20-38-02"
TRAIN_OWNER = "2026-08-30_15-06-39"


def make_splits(videos: list[str], n_test: int, val_frac: float, seed: int) -> dict:
    yt = sorted(v for v in videos if v.startswith("yt_"))
    rng = random.Random(seed)
    rng.shuffle(yt)
    test = sorted(yt[:n_test])
    n_val = max(1, round(val_frac * (len(yt) - n_test)))
    val = sorted(yt[n_test : n_test + n_val])
    train = sorted(yt[n_test + n_val :])
    return {
        "seed": seed,
        "note": "split by video; 2026-09-06 is test-only (never train/val)",
        "train": train + [TRAIN_OWNER],
        "val": val + [VAL_OWNER],
        "test": test + [TEST_OWNER],
    }


def load_splits(videos: list[str], a) -> dict:
    path = a.splits
    if path.exists() and not a.new_splits:
        sp = json.loads(path.read_text())
    elif path != SPLITS and SPLITS.exists() and not a.new_splits:
        sp = json.loads(SPLITS.read_text())  # a new split file starts from v1's assignment (same val/test clips)
    else:
        sp = make_splits(videos, a.n_test, a.val_frac, a.seed)
    from gro.auto.manifest import eval_ids

    known = {v for k in ("train", "val", "test") for v in sp[k]}
    new = sorted(v for v in videos if v not in known)
    if new:  # clips labelled after the split was drawn: ~val_frac to val, the rest to train (eval clips to test)
        rng = random.Random(a.seed + 1)
        ev = eval_ids()
        for v in new:
            if v.removeprefix("yt_") in ev or v in ev:
                sp["test"].append(v)
            else:
                sp["val" if rng.random() < a.val_frac else "train"].append(v)
        sp["note"] = sp.get("note", "") + f"; +{len(new)} clips assigned after the first draw"
    if new or not path.exists() or a.new_splits:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sp, indent=1) + "\n")
    assert TEST_OWNER not in sp["train"] + sp["val"], "2026-09-06 must stay out of train/val"
    leak = {v.removeprefix("yt_") for v in sp["train"] + sp["val"]} & eval_ids()
    assert not leak, f"eval clips (data_manifests split test/eval_*) in train/val: {sorted(leak)}"
    return sp


def wire_points(kp: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(bass, treble) px of wire f from a (50, 3) x, y, v array, or None if not labelled on both sides."""
    b, t = kp[2 * f], kp[2 * f + 1]
    return (b[:2], t[:2]) if b[2] > 0 and t[2] > 0 else None


def paint_wrap(img: np.ndarray, kp: np.ndarray, rng: random.Random) -> bool:
    """Synthetic fret wrap (D4 footage): a dark band across the neck between the nut and fret 1, extending past the
    board edges; the nut keypoints become occluded (vis 1, still supervised). False if the frets aren't labelled."""
    w1, w2 = wire_points(kp, 1), wire_points(kp, 2)
    if w1 is None or w2 is None:
        return False
    w0 = wire_points(kp, 0) or tuple(p1 + (p1 - p2) * 2 ** (1 / 12) for p1, p2 in zip(w1, w2, strict=True))
    c0, c1 = rng.uniform(0.15, 0.45), rng.uniform(0.6, 0.85)  # band between nut (0) and fret 1 (1)
    ext = rng.uniform(0.08, 0.25)  # beyond the outer strings
    quad = []
    for c, sides in ((c0, (0, 1)), (c1, (1, 0))):
        for s in sides:
            p = w0[s] + c * (w1[s] - w0[s])
            q = w0[1 - s] + c * (w1[1 - s] - w0[1 - s])
            quad.append(p + ext * (p - q))
    poly = np.round(np.array(quad)).astype(np.int32)
    h, w = img.shape[:2]
    if not ((poly[:, 0] >= 0) & (poly[:, 0] < w) & (poly[:, 1] >= 0) & (poly[:, 1] < h)).any():
        return False
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    base = (
        np.array([rng.uniform(8, 45)] * 3) if rng.random() < 0.8 else np.array([rng.uniform(20, 200) for _ in range(3)])
    )
    tex = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), (h, w, 1))
    layer = np.clip(base[None, None, :] + tex, 0, 255)
    m = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] / 255.0
    img[:] = (img * (1 - m) + layer * m).astype(np.uint8)
    for i in (0, 1):
        if kp[i, 2] > 0:
            kp[i, 2] = 1
    return True


def paint_fingers(img: np.ndarray, kp: np.ndarray, rng: random.Random) -> bool:
    """1-3 skin-toned finger-like ellipses across the neck at random labelled frets; covered keypoints -> vis 1."""
    frets = [f for f in range(1, 24) if wire_points(kp, f) is not None and wire_points(kp, f + 1) is not None]
    if not frets:
        return False
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for f in rng.sample(frets, min(len(frets), rng.randint(1, 3))):
        (b0, t0), (b1, t1) = wire_points(kp, f), wire_points(kp, f + 1)
        c = rng.uniform(0.2, 0.8)
        b, t = b0 + c * (b1 - b0), t0 + c * (t1 - t0)
        s0, s1 = rng.uniform(-0.3, 0.4), rng.uniform(0.5, 1.3)  # finger spans part of the neck from the bass side
        p, q = b + s0 * (t - b), b + s1 * (t - b)
        width = np.linalg.norm(b1 - b0) * rng.uniform(0.35, 0.7)
        ctr = (p + q) / 2
        ang = np.degrees(np.arctan2(*(q - p)[::-1]))
        axes = (int(np.linalg.norm(q - p) / 2) + 1, int(width / 2) + 1)
        cv2.ellipse(mask, (int(ctr[0]), int(ctr[1])), axes, ang, 0, 360, 255, -1, cv2.LINE_AA)
    skin = np.array([rng.uniform(60, 150), rng.uniform(90, 175), rng.uniform(130, 230)])  # BGR
    shade = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12, (h, w)), (0, 0), 3)
    layer = np.clip(skin[None, None, :] + shade[..., None], 0, 255)
    m = cv2.GaussianBlur(mask, (5, 5), 0)[..., None] / 255.0
    img[:] = (img * (1 - m) + layer * m).astype(np.uint8)
    for i, (x, y, v) in enumerate(kp):
        if v > 0 and 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)] > 127:
            kp[i, 2] = 1
    return True


def kpt_array(anchors: list[A.Anchor]) -> np.ndarray:
    kp = np.zeros((N_KPTS, 3))
    for a in anchors:
        if a.vis > 0 and 0 <= a.fret <= N_FRETS:
            kp[2 * a.fret + (a.side == "treble")] = (a.x, a.y, 2 if a.vis == 2 else 1)
    return kp


def yolo_line(anchors: list[A.Anchor], w: int, h: int, kp: np.ndarray | None = None) -> str | None:
    kp = kpt_array(anchors) if kp is None else kp
    lab = kp[kp[:, 2] > 0]
    if len(lab) < 4:
        return None
    # Pad the box by half a string spacing across the neck (the board extends past the outer strings) and a bit.
    by = {}
    for a in anchors:
        by.setdefault(a.fret, {})[a.side] = np.array([a.x, a.y])
    spans = [np.linalg.norm(d["treble"] - d["bass"]) for d in by.values() if len(d) == 2]
    pad = 0.1 * (max(spans) if spans else 0.0) + 0.02 * max(w, h)
    x0, y0 = lab[:, :2].min(0) - pad
    x1, y1 = lab[:, :2].max(0) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    vals = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for x, y, v in kp:
        # Keypoints outside the image are not labelled (YOLO would clip them anyway).
        if v == 0 or not (0 <= x < w and 0 <= y < h):
            vals += [0.0, 0.0, 0]
        else:
            vals += [x / w, y / h, int(v)]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, action="append", help="label dir(s) with frames.csv (default scale-v1)")
    ap.add_argument("--splits", type=Path, default=SPLITS, help="split file (created from scale-v1's if missing)")
    ap.add_argument(
        "--wrap-aug", type=float, default=0.0, help="fraction of train frames also exported with a fake fret wrap"
    )
    ap.add_argument(
        "--occ-aug", type=float, default=0.0, help="fraction of train frames also exported with fake fingers"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "datasets" / "scale-v1")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--n-test", type=int, default=5, help="held-out YouTube clips")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-splits", action="store_true", help="re-draw configs/scale-v1_splits.json")
    ap.add_argument("--max-per-split", type=int, default=0, help="cap frames per split (smoke tests)")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th kept frame per segment")
    ap.add_argument("--include-relative", action="store_true", help="also export non-anchorable frames")
    a = ap.parse_args()

    a.labels = a.labels or [ROOT / "data" / "labels" / "scale-v1"]
    rows = []
    label_dir = {}
    for d in a.labels:
        for r in csv.DictReader((d / "frames.csv").open()):
            rows.append(r)
            label_dir[r["seg_id"]] = d
    sp = load_splits(sorted({r["video"] for r in rows}), a)
    split_of = {v: s for s, vs in sp.items() if isinstance(vs, list) for v in vs}
    counts = Counter()
    jobs: dict[str, list[dict]] = defaultdict(list)  # seg_id -> rows to export
    for r in rows:
        s = split_of.get(r["video"])
        if s is None:
            counts["video_not_in_splits"] += 1
            continue
        if r["kept"] != "1":
            counts[f"{s}: dropped by filter"] += 1
            continue
        if r["anchorable"] != "1" and not a.include_relative:
            counts[f"{s}: not anchorable (skipped)"] += 1
            continue
        jobs[r["seg_id"]].append({**r, "split": s})
    rng = random.Random(a.seed)
    per_split = defaultdict(list)
    for seg, rs in jobs.items():
        for r in rs[:: a.stride]:
            per_split[r["split"]].append(r)
    if a.max_per_split:
        for s in per_split:
            rng.shuffle(per_split[s])
            per_split[s] = per_split[s][: a.max_per_split]
    todo = defaultdict(list)
    for s, rs in per_split.items():
        for r in rs:
            todo[r["seg_id"]].append(r)

    for s in ("train", "val", "test"):
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    manifest = []
    aug_rng = random.Random(a.seed + 7)
    for seg, rs in sorted(todo.items()):
        lf = A.load(label_dir[seg] / f"{seg}.json")
        by_idx = {int(f.extra["frame_index"]): f for f in lf.frames if "frame_index" in f.extra}
        want = sorted(int(r["frame_index"]) for r in rs)
        split_by = {int(r["frame_index"]): r["split"] for r in rs}
        cap = cv2.VideoCapture(str(ROOT / lf.video["path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
        idx = want[0]
        for fi in want:
            while idx < fi:  # sequential grab is exact (seeking can land on a keyframe for some codecs)
                cap.grab()
                idx += 1
            ok, img = cap.read()
            idx += 1
            if not ok:
                counts["read_failed"] += 1
                continue
            if lf.mirrored:
                img = cv2.flip(img, 1)  # canonical (D1); the labels are already canonical
            h, w = img.shape[:2]
            line = yolo_line(by_idx[fi].anchors, w, h)
            if line is None:
                counts["too_few_kpts"] += 1
                continue
            s = split_by[fi]
            variants = [("", img, None)]
            if s == "train":
                for tag, p, paint in (("wrap", a.wrap_aug, paint_wrap), ("occ", a.occ_aug, paint_fingers)):
                    if p > 0 and aug_rng.random() < p:
                        im2, kp2 = img.copy(), kpt_array(by_idx[fi].anchors)
                        if paint(im2, kp2, aug_rng):
                            if tag == "wrap" and aug_rng.random() < 0.5:
                                paint_fingers(im2, kp2, aug_rng)
                            variants.append((f"_{tag}", im2, kp2))
            for tag, im, kp in variants:
                line = yolo_line(by_idx[fi].anchors, w, h, kp)
                if line is None:
                    continue
                k = min(1.0, a.max_side / max(h, w))
                if k < 1:
                    im = cv2.resize(im, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
                name = f"{seg}_{fi:06d}{tag}"
                cv2.imwrite(str(a.out / "images" / s / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (a.out / "labels" / s / f"{name}.txt").write_text(line + "\n")
                counts[f"{s}: exported{tag}"] += 1
                manifest.append({"name": name, "split": s, "seg_id": seg, "video": lf.video["path"], "frame_index": fi})
        cap.release()

    data = {
        "path": str(a.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "fretboard"},
        "kpt_shape": [N_KPTS, 3],
        "flip_idx": FLIP_IDX,
        "kpt_names": {0: KPT_NAMES},
    }
    import yaml  # ultralytics dependency

    (a.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    with (a.out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["name", "split", "seg_id", "video", "frame_index"])
        w.writeheader()
        w.writerows(manifest)
    (a.out / "export_counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=1))
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")
    print(f"-> {a.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
