#!/bin/bash
#SBATCH --job-name=agsql-train
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
# Train an adapter into $AGSQL_ADAPTERS/$ADAPTER_NAME (default sql_adapter_2b).
# Extra args pass through to scripts/train_adapter.py:
#   ADAPTER_NAME=sql_adapter_2b_e1h sbatch hpc/train.sbatch --epochs 1 --val-frac 0.15
# Save the args first: sciclone.bashrc runs `set nonomatch` (a tcsh habit),
# which in bash overwrites "$@" with the word "nonomatch".
ARGS=("$@")
[ -f /usr/local/etc/sciclone.bashrc ] && source /usr/local/etc/sciclone.bashrc
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/train_adapter.py --out "$AGSQL_ADAPTERS/${ADAPTER_NAME:-sql_adapter_2b}" "${ARGS[@]}"
