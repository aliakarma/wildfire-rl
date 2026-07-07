#!/usr/bin/env bash
# Phase 5 (peer-review remediation) — generalization + scale suites, 3 parallel lanes.
set -u
cd "/mnt/c/Users/Ali Akarma/Documents/GitHub/wildfire-rl"
PY=~/venvs/wildfire-marl/bin/python
mkdir -p results/phase5_peer/logs

( $PY scripts/run_generalization_v2.py --region saudi \
    > results/phase5_peer/logs/gen_saudi.log 2>&1
  echo "GEN-SAUDI exit $?" ) &
( $PY scripts/run_generalization_v2.py --region california \
    > results/phase5_peer/logs/gen_california.log 2>&1
  echo "GEN-CALIFORNIA exit $?" ) &
( $PY scripts/run_scale_study.py --region saudi \
    > results/phase5_peer/logs/scale_saudi.log 2>&1
  echo "SCALE-SAUDI exit $?"
  $PY scripts/run_scale_study.py --region california \
    > results/phase5_peer/logs/scale_california.log 2>&1
  echo "SCALE-CALIFORNIA exit $?" ) &
wait
echo "PHASE5 SUITES COMPLETE"
