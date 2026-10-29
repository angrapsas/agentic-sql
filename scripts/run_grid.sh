#!/usr/bin/env bash
# The whole grid, cheap arms first. Safe to rerun: each arm resumes from its
# per-question checkpoint. Expect roughly 1.5 hours on a Colab T4 with a 2b model.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python data/verify_dataset.py
python -m agsql.cli run --arm all
python -m agsql.cli report
