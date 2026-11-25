#!/bin/bash
# Renamed-schema run on the cluster, unattended:
#   1. build data/Chinook_renamed.db (seconds)
#   2. train adapters on train_v1_renamed.jsonl: SIZES x 3 seeds
#   3. smoke, then every local arm in configs/renamed.yaml for those sizes
# Haiku/Sonnet arms run from a laptop (API), not here.
#
#   bash hpc/renamed.sh                 # 2b, 4b, 9b
#   SIZES="2b 4b" bash hpc/renamed.sh   # e.g. on astral, where 9b won't train
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export AGSQL_GRID=configs/renamed.yaml
export AGSQL_RESULTS=/sciclone/data10/$USER/agsql/results/renamed
export AGSQL_DB=$PWD/data/Chinook_renamed.db
(source hpc/env.sh >/dev/null 2>&1; python data/build_renamed.py db)

SIZES=${SIZES:-"2b 4b 9b"}
declare -A MODEL=([2b]=Qwen/Qwen3.5-2B [4b]=Qwen/Qwen3.5-4B [9b]=Qwen/Qwen3.5-9B)
ARMS=""
for tag in $SIZES; do
  [ -n "${MODEL[$tag]:-}" ] || { echo "unknown size: $tag"; exit 1; }
  ARMS="$ARMS,rn-$tag-oneshot,rn-$tag-loop,rn-$tag-loop-tuned-s0,rn-$tag-loop-tuned-s1,rn-$tag-loop-tuned-s2"
done
export ARMS=${ARMS#,}
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)

DEPS=""
for tag in $SIZES; do
  case $tag in 2b) T=0:20:00 ;; 4b) T=0:30:00 ;; *) T=0:50:00 ;; esac
  for seed in 0 1 2; do
    id=$(ADAPTER_NAME=renamed_${tag}_s${seed} sbatch --parsable --job-name=rn-${tag}-s${seed} --time=$T \
         hpc/train.sbatch --model-id "${MODEL[$tag]}" --train data/train_v1_renamed.jsonl \
         --seed $seed --val-frac 0.15 --epochs 3)
    DEPS="$DEPS:$id"
  done
done
S=$(sbatch --parsable --time=0:45:00 --dependency=afterok$DEPS hpc/smoke.sbatch)
G=$(sbatch --parsable --time=3:00:00 --dependency=afterok:$S --array=0-$((N - 1))%4 hpc/grid.sbatch)
echo "sizes: $SIZES | training, then smoke $S, then grid $G ($N arms)"
echo "results: AGSQL_GRID=$AGSQL_GRID AGSQL_DB=$AGSQL_DB AGSQL_RESULTS=$AGSQL_RESULTS python -m agsql.cli report"
