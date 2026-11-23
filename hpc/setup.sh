#!/bin/bash
# Run ONCE on gulf.sciclone.wm.edu (has internet; compute nodes may not).
# Installs packages and downloads weights only, no GPU work, which is what the
# front-end is for. Builds the env, downloads every model the grid uses, builds
# Chinook.db, runs the CPU tests. ~15 min, mostly torch and the downloads.
#
#   bash hpc/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK" "$AGSQL_RESULTS" "$AGSQL_ADAPTERS" "$HF_HOME" "$CONDA_PKGS_DIRS"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"
python -m pip install -r "$REPO/requirements.txt" anthropic pytest

# every model any arm or the training script loads, read from the grid
MODELS=$(python - <<'PY'
import yaml
g = yaml.safe_load(open("configs/grid.yaml"))
ids = {a["backend_args"]["model_id"] for a in g["arms"].values()
       if a.get("backend") == "hf" and not a.get("skip")}
print(" ".join(sorted(ids)))
PY
)
cd "$REPO"
for m in $MODELS; do
  echo "downloading $m"
  hf download "$m" >/dev/null
done

python data/verify_dataset.py
python -m pytest -q tests

echo
echo "done. next: sbatch hpc/smoke.sbatch"
