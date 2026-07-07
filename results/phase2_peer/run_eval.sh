#!/usr/bin/env bash
# Phase 2 (peer-review remediation) — certified-protocol evaluation, both regions parallel.
set -u
source ~/venvs/wildfire-marl/bin/activate
cd "/mnt/c/Users/Ali Akarma/Documents/GitHub/wildfire-rl"
mkdir -p results/phase2_peer/logs

python scripts/run_multiseed_eval_v2.py --region saudi \
  > results/phase2_peer/logs/eval_saudi.log 2>&1 &
python scripts/run_multiseed_eval_v2.py --region california \
  > results/phase2_peer/logs/eval_california.log 2>&1 &
wait
echo "EVAL COMPLETE"
tail -2 results/phase2_peer/logs/eval_saudi.log results/phase2_peer/logs/eval_california.log
