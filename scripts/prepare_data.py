#!/usr/bin/env python3
"""Split raw CSV into train/val/test and save dataset_meta.json."""
from k259016_assignment2.config import PROCESSED_DIR, RAW_DATA_CSV, SplitConfig
from k259016_assignment2.data import prepare_and_split


def main():
    paths = prepare_and_split(RAW_DATA_CSV, PROCESSED_DIR, SplitConfig())
    print("Data prepared:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
