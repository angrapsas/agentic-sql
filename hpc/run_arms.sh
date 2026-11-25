#!/bin/bash
# Run any set of local (GPU) arms from a grid: smoke test, then the full run.
#
#   bash hpc/run_arms.sh ladder  v2-2b-schema,v2-4b-schema
#   bash hpc/run_arms.sh renamed rn-2b-schema,rn-4b-schema
#
# Picks the grid, the results folder and, for renamed, the renamed database.
# Arms whose adapters do not exist yet will fail the smoke test; train first.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 2 ] || { sed -n 2,8p "$0"; exit 1; }
mkdir -p logs
export AGSQL_GRID=configs/$1.yaml
export AGSQL_RESULTS=/sciclone/data10/$USER/agsql/results/$1
case $1 in
  renamed) export AGSQL_DB=$PWD/data/Chinook_renamed.db
           [ -f "$AGSQL_DB" ] || (source hpc/env.sh >/dev/null 2>&1; python data/build_renamed.py db) ;;
  ladder)  export AGSQL_DB=$PWD/data/Chinook.db ;;
  *) echo "grid must be ladder or renamed"; exit 1 ;;
esac
export ARMS=$2
N=$(source hpc/env.sh >/dev/null 2>&1; python hpc/arms.py -n | wc -l)
S=$(sbatch --parsable --time=0:45:00 hpc/smoke.sbatch)
G=$(sbatch --parsable --time=2:00:00 --dependency=afterok:$S --array=0-$((N - 1))%4 hpc/grid.sbatch)
echo "$1: smoke $S, then grid $G ($N arms: $ARMS)"
echo "results: AGSQL_GRID=$AGSQL_GRID AGSQL_DB=$AGSQL_DB AGSQL_RESULTS=$AGSQL_RESULTS python -m agsql.cli report"
