#!/usr/bin/env python3
"""Train and save model under artifacts/model/."""
from k259016_assignment2.config import DATASET_META_JSON, MODEL_DIR, TRAIN_CSV, VAL_CSV, TrainConfig
from k259016_assignment2.trainer import train


def main():
    summary = train(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        meta_json=DATASET_META_JSON,
        out_dir=MODEL_DIR,
        cfg=TrainConfig(),
    )
    print("Training complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
