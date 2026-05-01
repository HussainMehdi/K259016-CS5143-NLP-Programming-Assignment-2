from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_CSV = PROJECT_ROOT / "CS5143-NLP PA2 data_train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
DATASET_META_JSON = PROCESSED_DIR / "dataset_meta.json"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
EVAL_DIR = PROJECT_ROOT / "artifacts" / "eval"
REPORT_MD = EVAL_DIR / "report.md"

SEED = 42

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.2
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 16
NUM_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 4

REPORT_TOP_K_ERRORS = 25

# Raw `price` is in local currency per `country` (MYR / SGD / PHP). Multiply by this to get USD.
# Sourced from exchange-rates.org daily pages (web, May 2026): MYR/SGD/PHP use 2026-05-01 quotes;
# no separate 2026-05-02 USD/MYR row was listed; MYR page notes rate unchanged since 2026-05-01.
# Implied: 1 MYR = 1/3.9700 USD, 1 SGD = 1/1.2726 USD, 1 PHP = 1/61.2895 USD.
FX_RATES_AS_OF = "2026-05-02"
FX_LOCAL_CURRENCY_TO_USD: dict[str, float] = {
    "my": 1.0 / 3.9700,  # USD/MYR = 3.9700 on 2026-05-01
    "sg": 1.0 / 1.2726,  # USD/SGD = 1.2726 on 2026-05-01
    "ph": 1.0 / 61.2895,  # USD/PHP = 61.2895 on 2026-05-01
}


@dataclass
class SplitConfig:
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = SEED


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    seed: int = SEED
    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    dropout: float = DROPOUT
    grad_clip_norm: float = GRAD_CLIP_NORM
    num_workers: int = NUM_WORKERS
    label_smoothing: float = LABEL_SMOOTHING


RAW_COLS = [
    "country",
    "sku_id",
    "title",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "short_description",
    "price",
    "product_type",
    "Concise_label",
]

CATEGORICAL_COLS = [
    "country",
    "product_type",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
]
