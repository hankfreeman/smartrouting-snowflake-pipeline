#!/usr/bin/env python3
"""
Build 4 x 3-tower exports from the 4-tower custom model (drop one tower at a time).
Meta optimized by holdout KS. Then optionally rescore feb11 and report KS comparison.

Usage:
  python build_3tower_from_4tower_custom.py
  python build_3tower_from_4tower_custom.py --rescore-feb11  # also rescore and print KS table
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import RANDOM_SEED
from ml.splits import load_train_val_test_holdout
from ml.train_router import encode_df

import train_multitower_sale_4towers_custom as mt4

EXPORT_4TOWER = mt4.EXPORT_DIR
EXPORT_3TOWER_BASE = REPO_ROOT / "exports" / "multitower_sale_3towers_custom_drop"
TOWER_NAMES = ["age_gender", "bps_income", "demographic", "lead"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build 3-tower exports from 4-tower custom (drop one tower each).")
    parser.add_argument("--rescore-feb11", action="store_true", help="Rescore feb11 with each 3-tower model and print KS table")
    args = parser.parse_args()

    model_path = EXPORT_4TOWER / "model.pkl"
    if not model_path.exists():
        print(f"4-tower model not found: {model_path}. Run train_multitower_sale_4towers_custom.py first.")
        return 1

    print("Loading 4-tower model and data...")
    towers, _, _ = joblib.load(model_path)
    tower_names_loaded = [t[0] for t in towers]
    if tower_names_loaded != TOWER_NAMES:
        print(f"Expected towers {TOWER_NAMES}, got {tower_names_loaded}")
        return 1

    train_df, val_df, test_df, holdout_df = load_train_val_test_holdout()
    train_df = mt4._add_bucket_columns(train_df)
    val_df = mt4._add_bucket_columns(val_df)
    test_df = mt4._add_bucket_columns(test_df)
    holdout_df = mt4._add_bucket_columns(holdout_df)
    for col in ["STABILITY_BUCKET", "BPS_TIER", "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"]:
        if col not in train_df.columns:
            train_df[col] = "unknown"
            val_df[col] = "unknown"
            test_df[col] = "unknown"
            holdout_df[col] = "unknown"
    lookups = mt4._build_lookups_with_buckets(train_df)

    def get_tower_probas(df):
        return [t[1].predict_proba(encode_df(df, lookups, feature_list=t[2])[0])[:, 1] for t in towers]

    y_train = train_df["SALE_MADE_FLAG"].astype(int).values
    y_val = val_df["SALE_MADE_FLAG"].astype(int).values
    y_test = test_df["SALE_MADE_FLAG"].astype(int).values
    y_holdout = holdout_df["SALE_MADE_FLAG"].astype(int).values

    P_train = get_tower_probas(train_df)
    P_val = get_tower_probas(val_df)
    P_test = get_tower_probas(test_df)
    P_holdout = get_tower_probas(holdout_df)

    # Threshold from 4-tower (for ref when saving; we'll recompute per 3-tower if needed)
    thresh_path = EXPORT_4TOWER / "threshold.json"
    default_thresh = 0.45
    if thresh_path.exists():
        with open(thresh_path) as f:
            default_thresh = json.load(f).get("threshold", default_thresh)

    results_summary = []

    for drop_idx, drop_name in enumerate(TOWER_NAMES):
        idx_3 = [i for i in range(4) if i != drop_idx]
        export_dir = Path(str(EXPORT_3TOWER_BASE) + "_" + drop_name)
        export_dir.mkdir(parents=True, exist_ok=True)

        X_train_meta = np.column_stack([P_train[i] for i in idx_3])
        X_val_meta = np.column_stack([P_val[i] for i in idx_3])
        X_test_meta = np.column_stack([P_test[i] for i in idx_3])
        X_holdout_meta = np.column_stack([P_holdout[i] for i in idx_3])

        best_params, _ = mt4.optimize_meta_by_holdout_ks(
            X_train_meta, y_train, X_val_meta, y_val, X_test_meta, y_test, X_holdout_meta, y_holdout,
        )
        meta_params = {
            "max_iter": 2000, "random_state": RANDOM_SEED, "class_weight": "balanced",
            "C": best_params["C"], "solver": best_params["solver"], "penalty": best_params["penalty"],
        }
        if best_params.get("l1_ratio") is not None:
            meta_params["l1_ratio"] = best_params["l1_ratio"]
        meta = LogisticRegression(**meta_params)
        meta.fit(X_train_meta, y_train)

        new_towers = [towers[i] for i in idx_3]
        joblib.dump((new_towers, meta, False), export_dir / "model.pkl")
        with open(export_dir / "threshold.json", "w") as f:
            json.dump({"threshold": default_thresh}, f)
        lookups_dir = export_dir / "lookups"
        lookups_dir.mkdir(parents=True, exist_ok=True)
        for col in list(mt4.CATEGORICAL_COLS) + mt4.BUCKET_CATEGORICAL_COLS:
            if col not in lookups:
                continue
            with open(lookups_dir / f"{col}.json", "w", encoding="utf-8") as f:
                json.dump(lookups.get(col, {}), f, sort_keys=True)

        print(f"  Built 3-tower (drop {drop_name}): {export_dir}")
        results_summary.append(("3tower_drop_" + drop_name, str(export_dir)))

    print("\nAll 4 x 3-tower exports built.")

    if args.rescore_feb11:
        import subprocess
        from scipy.stats import ks_2samp
        import pandas as pd

        input_csv = REPO_ROOT / "feb11apitest_api_results.csv"
        if not input_csv.exists():
            print(f"feb11 input not found: {input_csv}. Skip rescore.")
            return 0

        # KS for 4-tower (already have results)
        ks_4tower = 0.1333  # from earlier run
        ks_results = [("4_tower (all)", ks_4tower, "feb11apitest_4towers_custom_results.csv")]

        for config_label, export_dir in results_summary:
            out_csv = REPO_ROOT / f"feb11apitest_{config_label}_results.csv"
            cmd = [
                sys.executable, "rescore_feb11_with_5tower_only.py",
                "--model-dir", export_dir,
                "--input", str(input_csv),
                "--output", str(out_csv),
            ]
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True)
            if not out_csv.exists():
                continue
            df = pd.read_csv(out_csv)
            mask = df["PREDICTION_PROBA_5tower"].notna()
            df_scored = df.loc[mask]
            if df_scored.empty:
                continue
            if "SALE_MADE_BINARY" in df_scored.columns:
                y = pd.to_numeric(df_scored["SALE_MADE_BINARY"], errors="coerce").fillna(0).astype(int)
            elif "SALE_MADE_FLAG" in df_scored.columns:
                flag = df_scored["SALE_MADE_FLAG"].astype(str).str.upper()
                y = ((flag == "Y") | (flag == "1")).astype(int)
            else:
                continue
            if (y == 1).sum() == 0 or (y == 0).sum() == 0:
                continue
            proba = df_scored["PREDICTION_PROBA_5tower"].astype(float)
            ks, _ = ks_2samp(proba[y == 1], proba[y == 0])
            ks_results.append((config_label, ks, out_csv.name))

        print("\n" + "=" * 70)
        print("KS ON FEB11 (4-tower vs 3-tower drop-one)")
        print("=" * 70)
        for label, ks, _ in ks_results:
            print(f"  {label:35s}  KS = {ks:.4f}")
        best_label = max(ks_results, key=lambda x: x[1])
        print(f"\n  Best: {best_label[0]}  KS = {best_label[1]:.4f}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
