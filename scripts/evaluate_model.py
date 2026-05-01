#!/usr/bin/env python3
"""Evaluate checkpoint on val and test; write CSVs and JSON metrics."""
from k259016_assignment2.config import EVAL_DIR, MODEL_DIR, TEST_CSV, VAL_CSV
from k259016_assignment2.trainer import evaluate_split


def main():
    val_metrics = evaluate_split(
        model_dir=MODEL_DIR,
        split_csv=VAL_CSV,
        split_name="val",
        output_dir=EVAL_DIR,
    )
    test_metrics = evaluate_split(
        model_dir=MODEL_DIR,
        split_csv=TEST_CSV,
        split_name="test",
        output_dir=EVAL_DIR,
    )
    print("Evaluation complete:")
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)


if __name__ == "__main__":
    main()
