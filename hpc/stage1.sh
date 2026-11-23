#!/bin/bash
# Stage 1, one command, runs unattended:
#   1. train two adapters on the same 85/15 split, 3 epochs and 1 epoch
#      (each writes train_report.json: val loss per epoch, execution accuracy
#      on train and held-out pairs, and which modules got LoRA)
#   2. smoke test the new arms once both finish
#   3. full run of the new arms once the smoke passes
# If a step fails, the jobs after it never start: squeue shows them as
# (DependencyNeverSatisfied). Clear them with: scancel -u $USER
#
#   bash hpc/stage1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# slowest first, so the 4-at-a-time throttle starts them earliest
ARMS="qwen2b-loop-think-1k,qwen2b-loop-tuned-think-1k,qwen2b-oneshot-tuned,qwen2b-oneshot-tuned-e3h,qwen2b-oneshot-tuned-e1h,qwen2b-loop-tuned-e3h,qwen2b-loop-tuned-e1h"
export ARMS
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)

T3=$(ADAPTER_NAME=sql_adapter_2b_e3h sbatch --parsable hpc/train.sbatch --val-frac 0.15 --epochs 3)
T1=$(ADAPTER_NAME=sql_adapter_2b_e1h sbatch --parsable hpc/train.sbatch --val-frac 0.15 --epochs 1)
S=$(sbatch --parsable --dependency=afterok:$T3:$T1 hpc/smoke.sbatch)
G=$(sbatch --parsable --dependency=afterok:$S --array=0-$((N - 1))%4 hpc/grid.sbatch)

echo "train 3ep: $T3   train 1ep: $T1   smoke: $S   grid ($N arms): $G"
echo "watch with: squeue -u \$USER"
echo "afterwards: cat \$AGSQL_ADAPTERS/sql_adapter_2b_e*h/train_report.json | head; python -m agsql.cli report"
