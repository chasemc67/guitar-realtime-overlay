#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
until ssh -o ConnectTimeout=10 pc 'wsl -d Ubuntu -- test -f /home/chase/gro/repo/eval/speedup/extended.done'; do sleep 30; done
ssh pc 'wsl -d Ubuntu -- bash -s' <<'RUN'
set -e
cd ~/gro/repo
cp runs/fretnet/speedup_extended_512/best.pt /mnt/c/Users/chase/speedup_results/extended_512.pt
cp eval/speedup/extended_512/selection.json /mnt/c/Users/chase/speedup_results/selection_extended_512.json
cp eval/speedup/extended_512/held.json /mnt/c/Users/chase/speedup_results/held_extended_512.json
cp runs/fretnet/speedup_extended_512/log.jsonl /mnt/c/Users/chase/speedup_results/train_extended_512.jsonl
cp runs/fretnet/speedup_extended_512/protocol.json /mnt/c/Users/chase/speedup_results/protocol_extended_512.json
cp eval/speedup/extended_choice.json /mnt/c/Users/chase/speedup_results/extended_choice.json
RUN
scp pc:speedup_results/selection_extended_512.json pc:speedup_results/held_extended_512.json pc:speedup_results/train_extended_512.jsonl pc:speedup_results/protocol_extended_512.json pc:speedup_results/extended_choice.json eval/speedup/
scp pc:speedup_results/extended_512.pt /tmp/speedup_weights/
scp /tmp/speedup_weights/extended_512.pt eval/speedup/selection_extended_512.json mini:gro-coreml/speedup/
ssh mini 'touch ~/gro-coreml/speedup/extended_512.ready'
