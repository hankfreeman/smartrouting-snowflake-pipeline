"""
Load train/val/test and holdout data for router optimization.
Expects build_training_splits.py to have written:
  training_tables/train_global.csv, val_global.csv, test_global.csv, holdout_10pct.csv
"""
import pandas as pd

from .config import (
    TABLES_DIR,
    FEATURE_COLS,
    TARGET_COL,
    GENDER_COL,
    LEAD_SOURCE_SEG_COL,
    AGE_SEG_COL,
)

TRAIN_FILE = "train_global.csv"
VAL_FILE = "val_global.csv"
TEST_FILE = "test_global.csv"
HOLDOUT_FILE = "holdout_10pct.csv"


def load_train_val_test_holdout():
    """
    Load train, val, test, and holdout dataframes.
    Returns (train_df, val_df, test_df, holdout_df).
    Raises FileNotFoundError if any file is missing.
    """
    train_path = TABLES_DIR / TRAIN_FILE
    val_path = TABLES_DIR / VAL_FILE
    test_path = TABLES_DIR / TEST_FILE
    holdout_path = TABLES_DIR / HOLDOUT_FILE
    for p in (train_path, val_path, test_path, holdout_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. Run build_training_splits.py first."
            )
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    holdout_df = pd.read_csv(holdout_path, low_memory=False)
    return train_df, val_df, test_df, holdout_df


def required_columns():
    """Columns that must be present in split CSVs."""
    return [TARGET_COL] + list(FEATURE_COLS) + [GENDER_COL, LEAD_SOURCE_SEG_COL, AGE_SEG_COL]
