"""Compare cached direct predictions on exactly the labelled targets used by tracking."""

# ruff: noqa: E402
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_v2 as ev


def main():
    out = ROOT.parent
    for path in out.glob("tracking_*.json"):
        tracking = json.loads(path.read_text())
        if "macro" not in tracking:
            continue
        name = path.stem.removeprefix("tracking_")
        reference = "r832_fp16" if name in ("r832", "r832_scaled") else name
        direct = json.loads((out / f"held_{reference}.json").read_text())["held"]["CPU_AND_NE"]["frames"]
        selected = {group: {r["item"] for r in rows} for group, rows in tracking["frames"].items()}
        rows = {group: [r for r in direct[group] if r["item"] in ids] for group, ids in selected.items()}
        assert all(len(rows[group]) == len(ids) for group, ids in selected.items())
        clips = defaultdict(list)
        for row in rows["eval_v2"]:
            clips[row["clip"]].append(row)
        report = dict(
            macro=ev.macro({c: ev.agg(rs) for c, rs in clips.items()}),
            honest_0906=ev.agg(rows["honest_0906"]),
            reference=reference,
            tracking=name,
            source="cached direct predictions; no new inference or selection",
            excluded=tracking["excluded_no_predecessor"],
        )
        (out / f"control_{name}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
