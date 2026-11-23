#!/bin/bash
# Size ladder, one command, runs unattended. Before the first run, download the
# new model sizes on the login node with:  bash hpc/setup.sh
#
#   1. train 12 adapters: 4 sizes x 3 seeds, each on its own 85/15 split
#   2. smoke test every ladder arm once all 12 exist
#   3. run all 17 arms on chinook_v2, fast ones first
# If a step fails, later jobs sit as (DependencyNeverSatisfied): scancel -u $USER
#
#   bash hpc/ladder.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export AGSQL_GRID=configs/ladder.yaml
export AGSQL_RESULTS=/sciclone/data10/$USER/agsql/results/ladder
unset ARMS
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)

DEPS=""
for spec in 0.8b:Qwen/Qwen3.5-0.8B 2b:Qwen/Qwen3.5-2B 4b:Qwen/Qwen3.5-4B 9b:Qwen/Qwen3.5-9B; do
  tag=${spec%%:*}; model=${spec#*:}
  # Short, realistic limits: the scheduler can only backfill a job into a gap
  # in other people's schedules if its limit fits the gap.
  case $tag in 0.8b|2b) T=0:20:00 ;; 4b) T=0:30:00 ;; *) T=0:50:00 ;; esac
  for seed in 0 1 2; do
    id=$(ADAPTER_NAME=ladder_${tag}_s${seed} sbatch --parsable --job-name=tr-${tag}-s${seed} --time=$T \
         hpc/train.sbatch --model-id "$model" --seed $seed --val-frac 0.15 --epochs 3)
    DEPS="$DEPS:$id"
  done
done
S=$(sbatch --parsable --time=0:45:00 --dependency=afterok$DEPS hpc/smoke.sbatch)
G=$(sbatch --parsable --time=3:00:00 --dependency=afterok:$S --array=0-$((N - 1))%4 hpc/grid.sbatch)

echo "12 training jobs, then smoke $S, then grid $G ($N arms)"
echo "watch:   squeue -u \$USER"
echo "results: AGSQL_GRID=configs/ladder.yaml AGSQL_RESULTS=$AGSQL_RESULTS python -m agsql.cli report"
