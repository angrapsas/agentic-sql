#!/bin/bash
#SBATCH --job-name=agsql-smoke
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o logs/%x-%j.out
# One question through every local arm. Always before a real run.
#   mkdir -p logs && sbatch hpc/smoke.sbatch
[ -f /usr/local/etc/sciclone.bashrc ] && source /usr/local/etc/sciclone.bashrc
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Which GPU we got, and whether something else is already on it. Astral has
# placed jobs on a GPU another process was using, which ends in out-of-memory.
echo "host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
python -m agsql.cli run --arm "$(python hpc/arms.py)" --smoke
