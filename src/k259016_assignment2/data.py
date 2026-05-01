import html
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_COLS,
    FX_LOCAL_CURRENCY_TO_USD,
    FX_RATES_AS_OF,
    RAW_COLS,
    SplitConfig,
)
from .utils import dump_json


def load_training_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=RAW_COLS)


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _clean_category_str(series: pd.Series) -> pd.Series:
    """fillna -> clean_text; empty or whitespace-only result -> 'NA' (single missing token)."""
    s = series.fillna("").map(clean_text)
    return s.mask(s.eq(""), "NA")


def _map_categories(df: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        m = mappings[col]
        out[f"{col}_id"] = out[col].astype(str).map(lambda x: m.get(x, m["<UNK>"])).astype(int)
    return out


def build_category_mappings(df_train: pd.DataFrame) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        values = sorted(df_train[col].astype(str).unique().tolist())
        mapping = {"<UNK>": 0}
        for idx, val in enumerate(values, start=1):
            mapping[val] = idx
        mappings[col] = mapping
    return mappings


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title"] = out["title"].fillna("").map(clean_text)
    out["short_description"] = out["short_description"].fillna("").map(clean_text)
    out["category_lvl_1"] = _clean_category_str(out["category_lvl_1"])
    out["category_lvl_2"] = _clean_category_str(out["category_lvl_2"])
    out["category_lvl_3"] = _clean_category_str(out["category_lvl_3"])
    out["product_type"] = _clean_category_str(out["product_type"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    out["price"] = out["price"].clip(lower=0.0)

    cc = out["country"].astype(str).str.strip().str.lower()
    fx = cc.map(FX_LOCAL_CURRENCY_TO_USD)
    if fx.isna().any():
        bad = sorted(out.loc[fx.isna(), "country"].astype(str).unique().tolist())
        raise ValueError(
            "Unknown country code(s) for USD conversion: "
            f"{bad}. Add rates under FX_LOCAL_CURRENCY_TO_USD in config.py."
        )
    out["price"] = out["price"] * fx

    out["title_word_count"] = out["title"].str.split().str.len().fillna(0).astype(float)
    out["desc_word_count"] = out["short_description"].str.split().str.len().fillna(0).astype(float)
    out["log_price"] = np.log1p(out["price"])  # price is USD after FX above

    out["text_input"] = (
        "TITLE: "
        + out["title"]
        + " [SEP] CAT: "
        + out["category_lvl_1"].astype(str)
        + " > "
        + out["category_lvl_2"].astype(str)
        + " > "
        + out["category_lvl_3"].astype(str)
        + " [SEP] DESC: "
        + out["short_description"]
    )
    return out


def prepare_and_split(
    csv_path: Path,
    output_dir: Path,
    split_cfg: SplitConfig,
) -> dict[str, Path]:
    df = load_training_csv(csv_path)
    df = build_features(df)
    df["Concise_label"] = df["Concise_label"].astype(int)

    strat_key = df["Concise_label"].astype(str) + "_" + df["country"].astype(str)
    train_df, temp_df = train_test_split(
        df,
        train_size=split_cfg.train_size,
        random_state=split_cfg.random_seed,
        stratify=strat_key,
    )

    temp_ratio = split_cfg.val_size / (split_cfg.val_size + split_cfg.test_size)
    temp_strat = temp_df["Concise_label"].astype(str) + "_" + temp_df["country"].astype(str)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=temp_ratio,
        random_state=split_cfg.random_seed,
        stratify=temp_strat,
    )

    mappings = build_category_mappings(train_df)
    train_df = _map_categories(train_df, mappings)
    val_df = _map_categories(val_df, mappings)
    test_df = _map_categories(test_df, mappings)

    numeric_cols = ["title_word_count", "desc_word_count", "log_price"]
    mean_std: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        std = std if std > 1e-8 else 1.0
        mean_std[col] = {"mean": mean, "std": std}
        for frame in (train_df, val_df, test_df):
            frame[f"{col}_z"] = ((frame[col] - mean) / std).astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta_path = output_dir / "dataset_meta.json"
    dump_json(
        meta_path,
        {
            "split_config": asdict(split_cfg),
            "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "label_rate": {
                "train_positive_rate": float(train_df["Concise_label"].mean()),
                "val_positive_rate": float(val_df["Concise_label"].mean()),
                "test_positive_rate": float(test_df["Concise_label"].mean()),
            },
            "categorical_mappings": mappings,
            "numeric_stats": mean_std,
            "price_in_usd": True,
            "fx_rates_as_of": FX_RATES_AS_OF,
            "fx_local_currency_to_usd": FX_LOCAL_CURRENCY_TO_USD,
        },
    )

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "meta": meta_path,
    }

