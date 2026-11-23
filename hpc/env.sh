# Sourced by every job and by setup.sh. Edit WORK once, nothing else should need it.
#
# Home is small and quota'd: code only. Everything large (the conda env, model
# weights, adapters, results) lives under WORK. Check where your data dir is with
#   ls -l ~
# (the symlinks there point at your data and scratch directories).

export WORK="${WORK:-/sciclone/data10/$USER/agsql}"

export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src"
export AGSQL_RESULTS="${AGSQL_RESULTS:-$WORK/results}"
export AGSQL_ADAPTERS="$WORK/adapters"
export HF_HOME="$WORK/hf"
export CONDA_PKGS_DIRS="$WORK/conda-pkgs"
ENV_PREFIX="$WORK/env"

# Most W&M accounts default to tcsh, and `module` doesn't exist in a plain bash
# until this is sourced. Never run `conda init` on the cluster (RC's advice).
type module >/dev/null 2>&1 || source /usr/local/etc/sciclone.bashrc
module load miniforge3
eval "$(conda shell.bash hook)"
if [ -d "$ENV_PREFIX" ]; then conda activate "$ENV_PREFIX"; fi
