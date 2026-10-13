#!/usr/bin/env bash
# Collect only completed, VAL-frozen runs. Machine aliases follow _briefs/speedup.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /tmp/speedup_weights eval/speedup
for candidate in distill_resnet18_640 distill_resnet34_640 distill_mobilenet_v3_large_640 distill_efficientnet_b0_640 ft_640 ft_512 ft_416; do
  run="speedup_${candidate#distill_}"
  size="${candidate##*_}"
  until ssh -o ConnectTimeout=10 pc "wsl -d Ubuntu -- test -f /home/chase/gro/repo/eval/speedup/$candidate/selection.json"; do sleep 30; done
  echo "Collecting $candidate (selection frozen)"
  ssh pc 'wsl -d Ubuntu -- bash -s' <<RUN
set -e
cd ~/gro/repo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/speedup_eval.py --weights runs/fretnet/$run/best.pt --imgsz $size --out eval/speedup/$candidate --held-only > eval/speedup/held_$candidate.log 2>&1
mkdir -p /mnt/c/Users/chase/speedup_results
cp runs/fretnet/$run/best.pt /mnt/c/Users/chase/speedup_results/$candidate.pt
cp eval/speedup/$candidate/selection.json /mnt/c/Users/chase/speedup_results/selection_$candidate.json
cp eval/speedup/$candidate/held.json /mnt/c/Users/chase/speedup_results/held_$candidate.json
cp runs/fretnet/$run/log.jsonl /mnt/c/Users/chase/speedup_results/train_$candidate.jsonl
cp runs/fretnet/$run/protocol.json /mnt/c/Users/chase/speedup_results/protocol_$candidate.json
RUN
  scp "pc:speedup_results/selection_$candidate.json" "pc:speedup_results/held_$candidate.json" "pc:speedup_results/train_$candidate.jsonl" "pc:speedup_results/protocol_$candidate.json" eval/speedup/
  scp "pc:speedup_results/$candidate.pt" /tmp/speedup_weights/
  scp "/tmp/speedup_weights/$candidate.pt" "eval/speedup/selection_$candidate.json" mini:gro-coreml/speedup/
  ssh mini "touch ~/gro-coreml/speedup/$candidate.ready"
  echo "Transferred $candidate"
done
