#!/bin/bash
#SBATCH --job-name=agsql
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
# One array task per arm, so arms run in parallel on separate GPUs.
# Don't sbatch this directly, use hpc/submit.sh, which sizes the array.
[ -f /usr/local/etc/sciclone.bashrc ] && source /usr/local/etc/sciclone.bashrc
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ARM=$(python hpc/arms.py -n | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "task $SLURM_ARRAY_TASK_ID -> $ARM on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
python -m agsql.cli run --arm "$ARM"
