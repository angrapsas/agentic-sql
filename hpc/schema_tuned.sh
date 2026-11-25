#!/bin/bash
# Adapters trained on the full-schema prompt, then evaluated with it.
#   1. train SIZES x 3 seeds with train_adapter.py --prompt schema
#   2. smoke, then the full-schema tuned arms
# Run on gulf: astral's shared GPUs make training fail with out-of-memory.
#
#   bash hpc/schema_tuned.sh ladder            # normal Chinook, 4b and 9b
#   bash hpc/schema_tuned.sh renamed           # renamed Chinook, trained on renamed pairs
#   SIZES="4b" bash hpc/schema_tuned.sh ladder
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 1 ] || { sed -n 2,10p "$0"; exit 1; }
mkdir -p logs
export AGSQL_GRID=configs/$1.yaml
export AGSQL_RESULTS=/sciclone/data10/$USER/agsql/results/$1
case $1 in
  ladder)  export AGSQL_DB=$PWD/data/Chinook.db; TRAIN=data/train_v1.jsonl; ADP=ladder_schema; P=v2 ;;
  renamed) export AGSQL_DB=$PWD/data/Chinook_renamed.db; TRAIN=data/train_v1_renamed.jsonl; ADP=renamed_schema; P=rn
           [ -f "$AGSQL_DB" ] || (source hpc/env.sh >/dev/null 2>&1; python data/build_renamed.py db) ;;
  *) echo "grid must be ladder or renamed"; exit 1 ;;
esac
SIZES=${SIZES:-"4b 9b"}
declare -A MODEL=([4b]=Qwen/Qwen3.5-4B [9b]=Qwen/Qwen3.5-9B)
ARMS=""
for tag in $SIZES; do
  [ -n "${MODEL[$tag]:-}" ] || { echo "unknown size: $tag"; exit 1; }
  ARMS="$ARMS,$P-$tag-schema-tuned-s0,$P-$tag-schema-tuned-s1,$P-$tag-schema-tuned-s2"
done
export ARMS=${ARMS#,}
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)

DEPS=""
for tag in $SIZES; do
  case $tag in 4b) T=0:40:00 ;; *) T=1:15:00 ;; esac
  for seed in 0 1 2; do
    id=$(ADAPTER_NAME=${ADP}_${tag}_s${seed} sbatch --parsable --job-name=sc-${tag}-s${seed} --time=$T \
         hpc/train.sbatch --model-id "${MODEL[$tag]}" --train $TRAIN --prompt schema \
         --seed $seed --val-frac 0.15 --epochs 3)
    DEPS="$DEPS:$id"
  done
done
S=$(sbatch --parsable --time=0:45:00 --dependency=afterok$DEPS hpc/smoke.sbatch)
G=$(sbatch --parsable --time=1:00:00 --dependency=afterok:$S --array=0-$((N - 1))%4 hpc/grid.sbatch)
echo "$1, sizes $SIZES | training, then smoke $S, then grid $G ($N arms)"
echo "results: AGSQL_GRID=$AGSQL_GRID AGSQL_DB=$AGSQL_DB AGSQL_RESULTS=$AGSQL_RESULTS python -m agsql.cli report"
