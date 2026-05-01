#!/usr/bin/env python3
"""Build report.md from test predictions and metrics."""
import json

import pandas as pd

from k259016_assignment2.config import EVAL_DIR, REPORT_MD, REPORT_TOP_K_ERRORS


def main():
    split = "test"
    pred_path = EVAL_DIR / f"{split}_predictions.csv"
    metrics_path = EVAL_DIR / f"{split}_metrics.json"
    if not pred_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run evaluate_model.py first to generate prediction and metrics files.")

    preds = pd.read_csv(pred_path)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    preds["abs_error"] = (preds["Concise_label"] - preds["pred_prob_cal"]).abs()
    hard = preds.sort_values("abs_error", ascending=False).head(REPORT_TOP_K_ERRORS)

    rows = []
    rows.append(f"# Test set — conciseness ({split})")
    rows.append("")
    rows.append("## Metrics (calibrated)")
    m = metrics["metrics_calibrated"]
    rows.append(f"- RMSE: {m['rmse']:.6f}")
    rows.append(f"- Accuracy: {m['accuracy']:.4f}")
    rows.append(f"- F1: {m['f1']:.4f}")
    rows.append(f"- ROC-AUC: {m['roc_auc']:.4f}")
    rows.append("")
    rows.append("## Top Error Cases")
    rows.append("```text")
    rows.append(
        hard[
            ["sku_id", "title", "Concise_label", "pred_prob_cal", "pred_label_cal", "abs_error"]
        ].to_string(index=False)
    )
    rows.append("```")
    rows.append("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote report: {REPORT_MD}")


if __name__ == "__main__":
    main()
