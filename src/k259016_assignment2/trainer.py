from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import CATEGORICAL_COLS, TrainConfig
from .model import ConcisenessClassifier
from .utils import dump_json, load_json, rmse, set_seed


def train_config_from_checkpoint(ckpt: dict) -> TrainConfig:
    merged = asdict(TrainConfig())
    merged.update(ckpt.get("train_config", {}))
    return TrainConfig(**merged)


class HFTokenizerAdapter:
    def __init__(self, model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode_batch(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def save(self, out_dir: Path) -> None:
        self.tokenizer.save_pretrained(out_dir)

    @classmethod
    def load(cls, model_dir: Path) -> HFTokenizerAdapter:
        obj = cls.__new__(cls)
        obj.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return obj


class ConcisenessDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: HFTokenizerAdapter, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.labels = self.frame["Concise_label"].astype(int).to_numpy()

        text = self.frame["text_input"].fillna("").astype(str).tolist()
        tokenized = tokenizer.encode_batch(text, max_length=max_length)
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

        self.cat_ids = {
            col: torch.tensor(self.frame[f"{col}_id"].astype(int).to_numpy(), dtype=torch.long)
            for col in CATEGORICAL_COLS
        }
        self.numeric = torch.tensor(
            self.frame[["title_word_count_z", "desc_word_count_z", "log_price_z"]].astype(float).to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "numeric": self.numeric[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        for col in CATEGORICAL_COLS:
            item[col] = self.cat_ids[col][idx]
        return item


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _make_loader(dataset: ConcisenessDataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        batch = _to_device(batch, device)
        cat_inputs = {col: batch[col] for col in CATEGORICAL_COLS}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            categorical_inputs=cat_inputs,
            numeric_features=batch["numeric"],
        )
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "rmse": rmse(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    temp = torch.nn.Parameter(torch.ones(1, device=device))
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temp.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temp.detach().cpu().item())


def train(
    train_csv: Path,
    val_csv: Path,
    meta_json: Path,
    out_dir: Path,
    cfg: TrainConfig,
) -> Dict[str, float]:
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    meta = load_json(meta_json)
    cat_maps = meta["categorical_mappings"]
    category_cardinalities = {col: len(cat_maps[col]) for col in CATEGORICAL_COLS}

    tokenizer = HFTokenizerAdapter(cfg.model_name)
    train_ds = ConcisenessDataset(train_df, tokenizer=tokenizer, max_length=cfg.max_length)
    val_ds = ConcisenessDataset(val_df, tokenizer=tokenizer, max_length=cfg.max_length)
    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConcisenessClassifier(
        model_name=cfg.model_name,
        category_cardinalities=category_cardinalities,
        num_numeric_features=3,
        dropout=cfg.dropout,
    ).to(device)

    labels_series = train_df["Concise_label"].astype(int)
    n_pos = int((labels_series == 1).sum())
    n_neg = int((labels_series == 0).sum())
    pos_weight_tensor = torch.tensor(float(n_neg) / float(max(n_pos, 1)), device=device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_rmse = math.inf
    best_epoch = -1
    bad_epochs = 0
    history = []
    best_state = None
    eps = cfg.label_smoothing

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False)
        for batch in progress:
            batch = _to_device(batch, device)
            cat_inputs = {col: batch[col] for col in CATEGORICAL_COLS}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    categorical_inputs=cat_inputs,
                    numeric_features=batch["numeric"],
                )
                targets = batch["label"] * (1.0 - eps) + 0.5 * eps
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            progress.set_postfix({"train_loss": f"{running_loss / (progress.n + 1):.4f}"})

        val_logits, val_labels = predict_logits(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        metrics = evaluate_binary(val_labels.astype(int), val_probs)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(epoch_row)

        if metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint was selected.")
    model.load_state_dict(best_state)

    val_logits, val_labels = predict_logits(model, val_loader, device)
    temp = fit_temperature(val_logits, val_labels, device=device)
    val_probs_cal = 1.0 / (1.0 + np.exp(-(val_logits / temp)))
    val_metrics_cal = evaluate_binary(val_labels.astype(int), val_probs_cal)

    model_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": asdict(cfg),
            "temperature": temp,
            "category_cardinalities": category_cardinalities,
        },
        model_path,
    )
    tokenizer.save(out_dir / "tokenizer")

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_rmse_raw": best_val_rmse,
        "best_val_rmse_calibrated": val_metrics_cal["rmse"],
        "val_metrics_calibrated": val_metrics_cal,
        "temperature": temp,
        "device": str(device),
        "train_config": asdict(cfg),
    }
    dump_json(out_dir / "train_summary.json", summary)
    return summary


def evaluate_split(
    model_dir: Path,
    split_csv: Path,
    split_name: str,
    output_dir: Path,
) -> dict[str, float]:
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu")
    cfg = train_config_from_checkpoint(ckpt)

    tokenizer = HFTokenizerAdapter.load(model_dir / "tokenizer")
    frame = pd.read_csv(split_csv)
    dataset = ConcisenessDataset(frame, tokenizer=tokenizer, max_length=cfg.max_length)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConcisenessClassifier(
        model_name=cfg.model_name,
        category_cardinalities=ckpt["category_cardinalities"],
        num_numeric_features=3,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logits, labels = predict_logits(model, loader, device=device)
    probs_raw = 1.0 / (1.0 + np.exp(-logits))
    metrics_raw = evaluate_binary(labels.astype(int), probs_raw)

    temp = float(ckpt.get("temperature", 1.0))
    probs_cal = 1.0 / (1.0 + np.exp(-(logits / temp)))
    metrics_cal = evaluate_binary(labels.astype(int), probs_cal)

    pred_df = frame[["sku_id", "title", "Concise_label"]].copy()
    pred_df["pred_prob_raw"] = probs_raw
    pred_df["pred_prob_cal"] = probs_cal
    pred_df["pred_label_cal"] = (probs_cal >= 0.5).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    summary = {
        "split": split_name,
        "size": int(len(frame)),
        "metrics_raw": metrics_raw,
        "metrics_calibrated": metrics_cal,
        "temperature": temp,
    }
    dump_json(output_dir / f"{split_name}_metrics.json", summary)
    return summary
