#!/usr/bin/env bash
# Phase 2 (peer-review remediation) — full baseline training.
# 3 parallel lanes (one per algorithm); each lane trains saudi then california.
set -u
source ~/venvs/wildfire-marl/bin/activate
cd "/mnt/c/Users/Ali Akarma/Documents/GitHub/wildfire-rl"
mkdir -p results/phase2_peer/logs

run() {
  local cfg="$1" tag="$2"
  echo "[$(date +%H:%M:%S)] START $tag"
  python -m wildfire_marl.train.marl_train --config "$cfg" \
    > "results/phase2_peer/logs/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $tag (exit $?)"
}

( run configs/marl/phase2/mappo_saudi.yaml mappo_saudi
  run configs/marl/phase2/mappo_california.yaml mappo_california ) &
( run configs/marl/phase2/qmix_saudi.yaml qmix_saudi
  run configs/marl/phase2/qmix_california.yaml qmix_california ) &
( run configs/marl/phase2/commnet_saudi.yaml commnet_saudi
  run configs/marl/phase2/commnet_california.yaml commnet_california ) &
wait
echo "ALL TRAINING COMPLETE"
ls -la results/phase2_peer/
