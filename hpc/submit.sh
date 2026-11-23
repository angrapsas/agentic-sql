#!/bin/bash
# Submit the grid as a job array, at most 4 arms at once (gulf has 8 GPUs, shared).
#   bash hpc/submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
mkdir -p logs
N=$(python hpc/arms.py -n | wc -l)
python hpc/arms.py -n | nl -v0
sbatch --array=0-$((N - 1))%4 hpc/grid.sbatch
