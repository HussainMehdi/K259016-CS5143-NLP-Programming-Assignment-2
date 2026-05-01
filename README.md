# K259016 — CS5143 Assignment 2 (title conciseness classifier)

```bash
cd K259016-CS5143-NLP-Programming-Assignment-2
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
export PYTHONPATH=src
./scripts/run_pipeline.sh
```

To wipe prepared data and `artifacts/` before a clean rerun: **`./scripts/clean_artifacts.sh`**.

Or run in order: `prepare_data.py` -> `train_model.py` -> `evaluate_model.py` -> `report_metrics.py`.

Settings: `src/k259016_assignment2/config.py`.

Outputs: `data/processed/`, `artifacts/model/`, `artifacts/eval/` (metrics, prediction CSVs, `report.md`).  
Charts: `artifacts/plots/` (`training_curves.png`, `val_test_metrics.png`) from **`plot_training_metrics.py`**.