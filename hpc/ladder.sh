#!/bin/bash
# Size ladder, one command, runs unattended. Before the first run, download the
# new model sizes on the login node with:  bash hpc/setup.sh
#
#   1. train 12 adapters: 4 sizes x 3 seeds, each on its own 85/15 split
#   2. smoke test every ladder arm once all 12 exist
#   3. run all 17 arms on chinook_v2, fast ones first
# If a step fails, later jobs sit as (DependencyNeverSatisfied): scancel -u $USER
#
#   bash hpc/ladder.sh                     # all four sizes
#   SIZES="0.8b 2b 4b" bash hpc/ladder.sh  # a subset, e.g. on astral (A30s, 24GB:
#                                          # too tight to train the 9b)
# Results from both clusters land in the same folder on data10.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export AGSQL_GRID=configs/ladder.yaml
export AGSQL_RESULTS=/sciclone/data10/$USER/agsql/results/ladder
SIZES=${SIZES:-"0.8b 2b 4b 9b"}
declare -A MODEL=([0.8b]=Qwen/Qwen3.5-0.8B [2b]=Qwen/Qwen3.5-2B [4b]=Qwen/Qwen3.5-4B [9b]=Qwen/Qwen3.5-9B)

ARMS=""
for tag in $SIZES; do
  [ -n "${MODEL[$tag]:-}" ] || { echo "unknown size: $tag"; exit 1; }
  ARMS="$ARMS,v2-$tag-loop,v2-$tag-loop-tuned-s0,v2-$tag-loop-tuned-s1,v2-$tag-loop-tuned-s2"
done
case " $SIZES " in *" 0.8b "*) ARMS="v2-no-model$ARMS" ;; *) ARMS="${ARMS#,}" ;; esac
export ARMS
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)

DEPS=""
for tag in $SIZES; do
  model=${MODEL[$tag]}
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

echo "sizes: $SIZES | training, then smoke $S, then grid $G ($N arms)"
echo "watch:   squeue -u \$USER"
echo "results: AGSQL_GRID=configs/ladder.yaml AGSQL_RESULTS=$AGSQL_RESULTS python -m agsql.cli report"
