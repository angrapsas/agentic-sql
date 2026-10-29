#!/usr/bin/env bash
# One question through every arm. Run this before any real run: it catches the
# broken-backend and broken-protocol failures in two minutes instead of an hour in.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python data/verify_dataset.py
python -m agsql.cli run --arm all --smoke
