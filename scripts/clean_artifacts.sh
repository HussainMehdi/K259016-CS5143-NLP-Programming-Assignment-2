#!/usr/bin/env bash
# Remove prepared splits + model/eval outputs so you can rerun from scratch.
# Does NOT delete the raw course CSV in the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "${ROOT}/data/processed" "${ROOT}/artifacts"
echo "Removed: ${ROOT}/data/processed"
echo "Removed: ${ROOT}/artifacts"
echo "Run: export PYTHONPATH=src && bash scripts/run_pipeline.sh"
