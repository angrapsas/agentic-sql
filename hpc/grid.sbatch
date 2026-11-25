#!/bin/bash
#SBATCH --job-name=agsql
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
# One array task per arm, so arms run in parallel on separate GPUs.
# Don't sbatch this directly, use hpc/submit.sh, which sizes the array.
[ -f /usr/local/etc/sciclone.bashrc ] && source /usr/local/etc/sciclone.bashrc
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Which GPU we got, and whether something else is already on it. Astral has
# placed jobs on a GPU another process was using, which ends in out-of-memory.
echo "host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

ARM=$(python hpc/arms.py -n | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "task $SLURM_ARRAY_TASK_ID -> $ARM on $(hostname)"
python -m agsql.cli run --arm "$ARM"
