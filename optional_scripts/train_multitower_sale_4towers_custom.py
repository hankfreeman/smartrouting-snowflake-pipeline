#!/usr/bin/env python3
"""
4-tower model (custom): (1) age and gender, (2) BPS and est HH income, (3) demographic (race, language, ethnicity), (4) lead.
Target: SALE_MADE_FLAG. Meta model (logistic regression) blended; meta hyperparameters optimized by holdout KS.
Requires build_training_splits.py. Exports to exports/multitower_sale_4towers_custom.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.splits import load_train_val_test_holdout
from ml.train_router import build_lookups_from_df, encode_df
from ml.config import (
    CATEGORICAL_COLS,
    EXPORTS_DIR,
    RANDOM_SEED,
    MODEL_PARAMS,
    MIN_SAMPLES_FOR_GRADIENT_BOOSTING,
)

# Tower 1: age and gender
FEATURES_AGE_GENDER = ["GENDER", "AGE"]

# Tower 2: BPS and estimated household income
FEATURES_BPS_INCOME = [
    "BPS", "BPS_TIER", "EST_HH_INCOME",
    "INCOME_10K_BUCKET", "INCOME_BAND",
]

# Tower 3: demographic (race, language, ethnicity) — no religion
FEATURES_DEMOGRAPHIC = ["RACE_V2", "LANGUAGE", "ETHNICITY_GROUP_V2"]

# Tower 4: lead (contactability)
FEATURES_LEAD = [
    "ROKU_FLAG",
    "PHONE_1_LINKAGE_SCORE", "PHONE_1_CONTACTABILITY_SCORE", "PHONE_1_USAGE_12_MONTH", "PHONE_1_USAGE_2_MONTH",
]

BUCKET_CATEGORICAL_COLS = [
    "AGE_5YR_BUCKET", "INCOME_10K_BUCKET",
    "BPS_TIER", "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET", "INCOME_BAND",
]
AGE_5YR_EDGES = list(range(0, 86, 5)) + [999]
AGE_5YR_LABELS = [f"{a}-{a+4}" for a in range(0, 85, 5)] + ["85+"]
INCOME_10K_EDGES_THOUSANDS = list(range(0, 251, 10)) + [1e6]
INCOME_10K_LABELS = [f"{a}_{a+10}" for a in range(0, 250, 10)] + ["250_plus"]
INCOME_10K_EDGES_DOLLARS = list(range(0, 251_000, 10_000)) + [1e9]

EXPORT_DIR = EXPORTS_DIR / "multitower_sale_4towers_custom"
SELECT_BY = "holdout_KS"  # maximize holdout KS for generalization


def _add_bucket_columns(df):
    age = pd.to_numeric(df["AGE"], errors="coerce").fillna(0).clip(upper=85)
    df = df.copy()
    df["AGE_5YR_BUCKET"] = pd.cut(
        age, bins=AGE_5YR_EDGES, labels=AGE_5YR_LABELS, include_lowest=True, right=False
    ).astype(str)
    inc_raw = pd.to_numeric(df["EST_HH_INCOME"], errors="coerce").fillna(0).clip(lower=0)
    if inc_raw.max() >= 2000:
        inc = inc_raw
        edges, labels = INCOME_10K_EDGES_DOLLARS, INCOME_10K_LABELS
    else:
        inc = inc_raw
        edges, labels = INCOME_10K_EDGES_THOUSANDS, INCOME_10K_LABELS
    df["INCOME_10K_BUCKET"] = pd.cut(
        inc, bins=edges, labels=labels[: len(edges) - 1], include_lowest=True, right=False
    ).astype(str)
    return df


def _build_lookups_with_buckets(train_df):
    lookups = build_lookups_from_df(train_df)
    for col in BUCKET_CATEGORICAL_COLS:
        if col not in train_df.columns:
            continue
        lookups[col] = {}
        for raw in train_df[col].astype(str).replace("nan", "").replace("None", "").unique():
            raw = str(raw).strip()
            if raw and raw not in lookups[col]:
                lookups[col][raw] = len(lookups[col]) + 1
    return lookups


def _fit_tower(X_train, y_train, verbose=True):
    n_train = len(y_train)
    n_pos, n_neg = int((y_train == 1).sum()), n_train - int((y_train == 1).sum())
    use_balanced = n_pos >= 30 and n_neg >= 30
    if n_train >= MIN_SAMPLES_FOR_GRADIENT_BOOSTING:
        if verbose:
            print(f"    GradientBoosting (n={n_train})")
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=4,
            learning_rate=0.1, subsample=0.8, random_state=RANDOM_SEED,
        )
        if use_balanced:
            model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(X_train, y_train)
    else:
        if verbose:
            print(f"    RandomForest (n={n_train})")
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)
    return model


def _fit_and_ks(meta_params, X_train_meta, y_train, X_val_meta, y_val, X_test_meta, y_test, X_holdout_meta, y_holdout):
    """Fit meta model and return (ks_val, ks_test, ks_holdout)."""
    try:
        meta = LogisticRegression(**meta_params)
        meta.fit(X_train_meta, y_train)
        out = []
        for X, y in [(X_val_meta, y_val), (X_test_meta, y_test)]:
            if X is None or y is None or len(y) == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
                out.append(None)
            else:
                p = meta.predict_proba(X)[:, 1]
                ks, _ = ks_2samp(p[y == 1], p[y == 0])
                out.append(ks)
        if X_holdout_meta is not None and y_holdout is not None and len(y_holdout) > 0 and (y_holdout == 1).sum() > 0 and (y_holdout == 0).sum() > 0:
            p = meta.predict_proba(X_holdout_meta)[:, 1]
            ks_h, _ = ks_2samp(p[y_holdout == 1], p[y_holdout == 0])
            out.append(ks_h)
        else:
            out.append(None)
        return tuple(out)
    except Exception:
        return None, None, None


def optimize_meta_by_holdout_ks(X_train_meta, y_train, X_val_meta, y_val, X_test_meta, y_test, X_holdout_meta, y_holdout):
    """Search meta hyperparameters; select best by holdout_KS (fallback: test_KS, then val_KS)."""
    C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    results = []

    def run_config(solver, penalty, C, l1_ratio=None):
        params = {
            "max_iter": 1000, "random_state": RANDOM_SEED, "class_weight": "balanced",
            "C": C, "solver": solver, "penalty": penalty,
        }
        if penalty == "elasticnet" and l1_ratio is not None:
            params["l1_ratio"] = l1_ratio
        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            return
        ks_val, ks_test, ks_holdout = _fit_and_ks(
            params, X_train_meta, y_train, X_val_meta, y_val,
            X_test_meta, y_test, X_holdout_meta, y_holdout,
        )
        if ks_val is None:
            return
        results.append({
            "solver": solver, "penalty": penalty, "C": C, "l1_ratio": l1_ratio,
            "val_KS": ks_val, "test_KS": ks_test, "holdout_KS": ks_holdout,
        })

    for solver in ["lbfgs", "liblinear", "saga"]:
        for C in C_VALUES:
            run_config(solver, "l2", C)
    for solver in ["liblinear", "saga"]:
        for C in C_VALUES:
            run_config(solver, "l1", C)
    for C in C_VALUES[:6]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            run_config("saga", "elasticnet", C, l1_ratio=l1_ratio)

    # Best by holdout_KS (NaN last), then test_KS, then val_KS
    for r in results:
        r["_select"] = r.get("holdout_KS") if r.get("holdout_KS") is not None else -1
    results_sorted = sorted(results, key=lambda x: x["_select"], reverse=True)
    if not results_sorted or results_sorted[0]["_select"] < 0:
        for r in results:
            r["_select"] = r.get("test_KS") if r.get("test_KS") is not None else -1
        results_sorted = sorted(results, key=lambda x: x["_select"], reverse=True)
    if not results_sorted or results_sorted[0]["_select"] < 0:
        results_sorted = sorted(results, key=lambda x: x["val_KS"], reverse=True)

    best = results_sorted[0]
    best_params = {
        "solver": best["solver"], "penalty": best["penalty"], "C": best["C"],
        "l1_ratio": best.get("l1_ratio"),
    }
    return best_params, results_sorted


def main():
    print("Loading train/val/test/holdout ...")
    train_df, val_df, test_df, holdout_df = load_train_val_test_holdout()
    train_df = _add_bucket_columns(train_df)
    val_df = _add_bucket_columns(val_df)
    test_df = _add_bucket_columns(test_df)
    holdout_df = _add_bucket_columns(holdout_df)
    for col in ["STABILITY_BUCKET", "BPS_TIER", "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"]:
        if col not in train_df.columns:
            train_df[col] = "unknown"
            val_df[col] = "unknown"
            test_df[col] = "unknown"
            holdout_df[col] = "unknown"
    lookups = _build_lookups_with_buckets(train_df)

    features_age_gender = [c for c in FEATURES_AGE_GENDER if c in train_df.columns]
    features_bps_income = [c for c in FEATURES_BPS_INCOME if c in train_df.columns]
    features_demographic = [c for c in FEATURES_DEMOGRAPHIC if c in train_df.columns]
    features_lead = [c for c in FEATURES_LEAD if c in train_df.columns]

    TOWER_NAMES = ["age_gender", "bps_income", "demographic", "lead"]
    TOWER_FEATURES = [features_age_gender, features_bps_income, features_demographic, features_lead]

    target = "SALE_MADE_FLAG"
    y_train = train_df[target].astype(int).values
    y_val = val_df[target].astype(int).values
    y_test = test_df[target].astype(int).values
    y_holdout = holdout_df[target].astype(int).values

    print("\n--- 4-tower (age_gender, bps_income, demographic, lead) for SALE_MADE_FLAG ---")
    towers = []
    P_train_list = []
    for name, feat_list in zip(TOWER_NAMES, TOWER_FEATURES):
        X_t, _ = encode_df(train_df, lookups, feature_list=feat_list)
        if X_t.shape[1] == 0:
            print(f"  Skip tower {name}: no features")
            continue
        t = _fit_tower(X_t, y_train, verbose=True)
        towers.append((name, t, feat_list))
        P_train_list.append(t.predict_proba(X_t)[:, 1])

    if len(towers) < 2:
        print("Need at least 2 towers.")
        return 1

    def get_tower_probas(df):
        return [t[1].predict_proba(encode_df(df, lookups, feature_list=t[2])[0])[:, 1] for t in towers]

    P_val_list = get_tower_probas(val_df)
    P_test_list = get_tower_probas(test_df)
    P_holdout_list = get_tower_probas(holdout_df)

    X_train_meta = np.column_stack(P_train_list)
    X_val_meta = np.column_stack(P_val_list)
    X_test_meta = np.column_stack(P_test_list)
    X_holdout_meta = np.column_stack(P_holdout_list)

    print("\n--- Meta-model: optimize by holdout KS ---")
    best_params, all_results = optimize_meta_by_holdout_ks(
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

    # Report best
    best_row = all_results[0]
    print(f"  Best by {SELECT_BY}: holdout_KS={best_row.get('holdout_KS')}, test_KS={best_row.get('test_KS')}, val_KS={best_row['val_KS']}")
    print(f"  Meta: {best_params['solver']} {best_params['penalty']} C={best_params['C']}" + (f" l1_ratio={best_params.get('l1_ratio')}" if best_params.get('l1_ratio') is not None else ""))

    # Threshold by val F1
    proba_val = meta.predict_proba(X_val_meta)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in [x / 100.0 for x in range(5, 76, 5)]:
        pred = (proba_val >= t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0.0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    proba_test = meta.predict_proba(X_test_meta)[:, 1]
    pred_test = (proba_test >= best_t).astype(int)
    test_f1 = f1_score(y_test, pred_test, zero_division=0.0)
    ks_test, _ = ks_2samp(proba_test[y_test == 1], proba_test[y_test == 0])
    print(f"  Threshold (val F1): {best_t:.2f}  Test F1: {test_f1:.4f}  Test KS: {ks_test:.4f}")

    import joblib
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((towers, meta, False), EXPORT_DIR / "model.pkl")
    with open(EXPORT_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)
    lookups_dir = EXPORT_DIR / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)
    for col in list(CATEGORICAL_COLS) + BUCKET_CATEGORICAL_COLS:
        if col not in lookups:
            continue
        with open(lookups_dir / f"{col}.json", "w", encoding="utf-8") as f:
            json.dump(lookups.get(col, {}), f, sort_keys=True)

    # Save optimization results for reference
    import csv
    results_path = REPO_ROOT / "ks_4towers_custom_holdout_results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["solver", "penalty", "C", "l1_ratio", "val_KS", "test_KS", "holdout_KS"])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"  Meta search results saved to {results_path}")
    print(f"  Model exported to {EXPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
