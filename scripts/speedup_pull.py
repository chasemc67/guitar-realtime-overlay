"""Fetch completed Mini results once, without racing partially written reports."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/speedup"
REMOTE = """
import json
from pathlib import Path
root=Path.home()/'gro-coreml/speedup'
files=[]
for p in root.glob('*.json'):
 try:
  s=json.loads(p.read_text())
 except (ValueError,OSError):
  continue
 valid=(p.name.startswith('held_') and 'ALL' in s.get('held',{}))
 valid |= p.name.startswith('tracking_') and 'macro' in s
 valid |= p.name.startswith('video_') and len(s.get('clips',[]))==3
 valid |= p.name.endswith('.compression.json') or p.name=='compressed_candidates.json'
 if valid:
  files.append(p.name)
  files.extend(q.name for q in root.glob(p.stem+'.*.npz'))
  if p.name.startswith('tracking_'):
   files.append(p.stem+'.selection.json')
print(json.dumps(files))
"""


def main():
    names = json.loads(subprocess.check_output(["ssh", "mini", "python3 -c " + shlex.quote(REMOTE)], text=True))
    for name in names:
        path = OUT / ("ml_" + name)
        if path.exists():
            continue
        temp = path.with_suffix(path.suffix + ".partial")
        subprocess.run(["scp", "-q", "mini:gro-coreml/speedup/" + name, str(temp)], check=True)
        temp.replace(path)
        print("Retrieved", name, flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/speedup_report.py")], check=True)


if __name__ == "__main__":
    main()
