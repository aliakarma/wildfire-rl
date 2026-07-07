#!/usr/bin/env bash
set -e
source ~/venvs/wildfire-marl/bin/activate
cd "/mnt/c/Users/Ali Akarma/Documents/GitHub/wildfire-rl"
python -c 'import torch, wildfire_marl; print("cuda:", torch.cuda.is_available())'
for algo in mappo qmix commnet; do
  echo "--- smoke $algo ---"
  python -m wildfire_marl.train.marl_train --config "results/phase2_peer/smoke/${algo}_smoke.yaml" 2>&1 | tail -4
done
ls -la results/phase2_peer/smoke/
