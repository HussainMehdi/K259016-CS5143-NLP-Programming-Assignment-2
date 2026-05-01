#!/usr/bin/env python3
"""Plot training history and val/test metrics to PNGs under artifacts/plots/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from k259016_assignment2.config import EVAL_DIR, MODEL_DIR

HISTORY_CSV = MODEL_DIR / "training_history.csv"
TRAIN_SUMMARY = MODEL_DIR / "train_summary.json"
VAL_METRICS_JSON = EVAL_DIR / "val_metrics.json"
TEST_METRICS_JSON = EVAL_DIR / "test_metrics.json"
PLOTS_DIR = MODEL_DIR.parent / "plots"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(df: pd.DataFrame, best_epoch: int | None, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Training / validation (per epoch)", fontsize=14)

    axes[0, 0].plot(df["epoch"], df["train_loss"], color="tab:blue", marker="o", markersize=3)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Train loss (BCE weighted)")
    axes[0, 0].grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_rmse"], color="tab:red", marker="o", markersize=3, label="Val RMSE")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.5, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1, 0].plot(df["epoch"], df["val_accuracy"], color="tab:green", marker="o", markersize=3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Val accuracy")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(df["epoch"], df["val_f1"], color="tab:purple", marker="o", markersize=3, label="F1")
    if "val_roc_auc" in df.columns and df["val_roc_auc"].notna().any():
        ax4.plot(df["epoch"], df["val_roc_auc"], color="tab:orange", marker="s", markersize=3, label="ROC-AUC")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0.0, 1.0)
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_test_bars(out_path: Path) -> None:
    val = _load_json(VAL_METRICS_JSON)
    test = _load_json(TEST_METRICS_JSON)
    if val is None:
        return

    keys = ["rmse", "accuracy", "f1", "roc_auc"]
    titles = ["RMSE (lower better)", "Accuracy", "F1", "ROC-AUC"]
    m_val = val.get("metrics_calibrated") or val.get("metrics_raw") or {}
    m_test = (test or {}).get("metrics_calibrated") or (test or {}).get("metrics_raw") or {}

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle("Validation vs test (calibrated if available)", fontsize=13)
    for ax, k, title in zip(axes.flat, keys, titles):
        v0 = float(m_val.get(k, np.nan))
        t_raw = m_test.get(k) if test else None
        t0 = float(t_raw) if t_raw is not None and str(t_raw) != "nan" else None
        if t0 is not None and not np.isnan(t0):
            ax.bar(["Val", "Test"], [v0, t0], color=["steelblue", "coral"], width=0.55)
        else:
            ax.bar(["Val"], [v0], color="steelblue", width=0.45)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_CSV.exists():
        raise FileNotFoundError(f"Missing {HISTORY_CSV}; run train_model.py first.")

    df = pd.read_csv(HISTORY_CSV)
    summary = _load_json(TRAIN_SUMMARY)
    best_epoch = summary.get("best_epoch") if summary else None

    curves_path = PLOTS_DIR / "training_curves.png"
    plot_training_history(df, best_epoch, curves_path)
    print(f"Wrote {curves_path}")

    bars_path = PLOTS_DIR / "val_test_metrics.png"
    if VAL_METRICS_JSON.exists():
        plot_val_test_bars(bars_path)
        print(f"Wrote {bars_path}")
    else:
        print(f"Skip bar chart: missing {VAL_METRICS_JSON} (run evaluate_model.py first)")


if __name__ == "__main__":
    main()
