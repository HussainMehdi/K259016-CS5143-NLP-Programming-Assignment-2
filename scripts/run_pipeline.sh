#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/report_metrics.py
python scripts/plot_training_metrics.py
