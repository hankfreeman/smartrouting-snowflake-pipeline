#!/usr/bin/env python3
"""
Build training_tables (train_global, val_global, test_global, holdout_10pct) from 22k.csv.
Use 22k for training; then use validatemarch.csv for testing/validation.

Usage:
  python build_training_splits_from_22k.py
  python build_training_splits_from_22k.py --input 22k.csv --seed 42

Writes:
  training_tables/train_global.csv  (60%)
  training_tables/val_global.csv    (20%)
  training_tables/test_global.csv  (10%)
  training_tables/holdout_10pct.csv (10%)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import (
    TABLES_DIR,
    RANDOM_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    HOLDOUT_FRAC,
    TARGET_COL,
    GENDER_COL,
    LEAD_SOURCE_SEG_COL,
    AGE_SEG_COL,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_sale_made_flag(df):
    if TARGET_COL not in df.columns:
        return df
    raw = df[TARGET_COL].astype(str).str.strip().str.upper()
    df = df.copy()
    df[TARGET_COL] = (raw.isin(("Y", "1", "TRUE", "YES")) | (df[TARGET_COL] == 1)).astype(int)
    return df


def _add_lead_source_seg(df):
    if LEAD_SOURCE_SEG_COL in df.columns:
        return df
    df = df.copy()
    df[LEAD_SOURCE_SEG_COL] = "Other"
    return df


def _add_age_seg(df):
    if AGE_SEG_COL in df.columns:
        return df
    df = df.copy()
    age = pd.to_numeric(df.get("AGE"), errors="coerce").fillna(0)
    seg = pd.Series("UNKNOWN", index=df.index)
    seg[age < 55] = "BELOW_55"
    seg[(age >= 55) & (age < 70)] = "55_70"
    seg[age >= 70] = "ABOVE_70"
    df[AGE_SEG_COL] = seg
    return df


def _add_placed_flag_if_missing(df):
    df = df.copy()
    if "PLACED_FLAG" not in df.columns:
        df["PLACED_FLAG"] = 0
    if "ANNUAL_PREMIUM" not in df.columns:
        df["ANNUAL_PREMIUM"] = np.nan
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build training splits from 22k.csv")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "22k.csv", help="Input CSV (22k)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    df = _normalize_sale_made_flag(df)
    df = _add_lead_source_seg(df)
    df = _add_age_seg(df)
    df = _add_placed_flag_if_missing(df)

    # Optional: run feature engineering so train/val/test have same columns as existing pipeline
    try:
        from ml.feature_engineering import add_engineered_features
        df, _, _ = add_engineered_features(df)
        print("  Applied feature engineering.")
    except Exception as e:
        print(f"  Note: feature_engineering skipped ({e}). Training may add columns on the fly.")

    np.random.seed(args.seed)

    # Holdout 10%
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    holdout_size = int(n * HOLDOUT_FRAC)
    holdout_idx = idx[:holdout_size]
    rest_idx = idx[holdout_size:]
    df_holdout = df.iloc[holdout_idx].copy()
    df_rest = df.iloc[rest_idx].copy()

    # Of the remaining 90%: train 60% of full = 66.67% of rest, val 20% of full = 22.22% of rest, test 10% of full = 11.11% of rest
    n_rest = len(df_rest)
    train_size = int(n_rest * (TRAIN_FRAC / (1 - HOLDOUT_FRAC)))
    val_size = int(n_rest * (VAL_FRAC / (1 - HOLDOUT_FRAC)))
    test_size = n_rest - train_size - val_size

    df_train = df_rest.iloc[:train_size].copy()
    df_val = df_rest.iloc[train_size : train_size + val_size].copy()
    df_test = df_rest.iloc[train_size + val_size :].copy()

    train_path = TABLES_DIR / "train_global.csv"
    val_path = TABLES_DIR / "val_global.csv"
    test_path = TABLES_DIR / "test_global.csv"
    holdout_path = TABLES_DIR / "holdout_10pct.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_holdout.to_csv(holdout_path, index=False)

    print(f"  Train:   {len(df_train):,} -> {train_path}")
    print(f"  Val:     {len(df_val):,} -> {val_path}")
    print(f"  Test:    {len(df_test):,} -> {test_path}")
    print(f"  Holdout: {len(df_holdout):,} -> {holdout_path}")
    print("Done. Next: python train_multitower_sale_5towers.py")


if __name__ == "__main__":
    main()
