#!/bin/bash
#SBATCH --job-name=agsql-smoke
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o logs/%x-%j.out
# One question through every local arm. Always before a real run.
#   mkdir -p logs && sbatch hpc/smoke.sbatch
[ -f /usr/local/etc/sciclone.bashrc ] && source /usr/local/etc/sciclone.bashrc
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m agsql.cli run --arm "$(python hpc/arms.py)" --smoke
