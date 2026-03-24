"""
Train all segments of a router: per-segment RFE (maximize validation F1), then fit final model.
Saves to exports/router_<router_id>/<segment_name>/model.pkl and selected_features.json,
and exports/router_<router_id>/lookups/*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    EXPORTS_DIR,
    LOOKUPS_DIR,
    RANDOM_SEED,
    MODEL_PARAMS,
    RFE_BASE_ESTIMATOR_PARAMS,
    RFE_N_FEATURES_MIN,
    RFE_N_FEATURES_MAX,
    RFE_STEP,
    MIN_SAMPLES_FOR_GRADIENT_BOOSTING,
)
from .routers import (
    get_segment_mask,
    get_router_spec,
    get_training_feature_cols,
    get_training_feature_cols_for_dims,
)
from .splits import load_train_val_test_holdout


def _log(msg="", end="\n"):
    """Print and flush so progress appears immediately."""
    print(msg, end=end)
    sys.stdout.flush()


def build_lookups_from_df(df):
    """Build label encoding {col: {raw_value: encoded_int}} from dataframe."""
    lookups = {col: {} for col in CATEGORICAL_COLS}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        for raw in df[col].astype(str).replace("nan", "").replace("None", "").unique():
            raw = str(raw).strip()
            if raw and raw not in lookups[col]:
                lookups[col][raw] = len(lookups[col]) + 1
    return lookups


def encode_df(df, lookups, feature_list=None):
    """Return (X, y) with X float matrix, y 0/1. feature_list defaults to FEATURE_COLS.
    Columns with a lookup (e.g. CATEGORICAL_COLS or bucket cols) use that; all others encoded as numeric (float)."""
    if feature_list is None:
        feature_list = FEATURE_COLS
    rows = []
    for _, r in df.iterrows():
        row = []
        for col in feature_list:
            raw = r.get(col, "")
            lookup = lookups.get(col, {})
            if lookup:
                key = str(raw).strip() if raw not in (None, "", "nan") else ""
                row.append(float(lookup.get(key, 0)))
            else:
                try:
                    v = float(raw) if raw not in (None, "", "nan") else 0.0
                    row.append(0.0 if np.isnan(v) else v)
                except (TypeError, ValueError):
                    row.append(0.0)
        rows.append(row)
    X = np.asarray(rows, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(df[TARGET_COL].astype(int), dtype=np.int32)
    return X, y


def rfe_by_val_f1(X_train, y_train, X_val, y_val, feature_list, rfe_min=None, rfe_max=None, verbose=False):
    """
    Try n_features in [rfe_min, rfe_max]; pick n and feature set that maximize validation F1.
    feature_list: column names for X_train (training features only, no routing dims).
    Returns (best_selected_features, best_validation_f1).
    """
    rfe_min = rfe_min or RFE_N_FEATURES_MIN
    rfe_max = rfe_max or RFE_N_FEATURES_MAX
    n_cols = len(feature_list)
    n_range = list(range(rfe_min, min(rfe_max, X_train.shape[1], n_cols) + 1))
    best_f1 = -1.0
    best_features = list(feature_list)[: min(n_cols, rfe_min)]
    for i, n in enumerate(n_range):
        if verbose:
            _log(f"      RFE n_features={n} ({i+1}/{len(n_range)}) ... ", end="")
        rfe = RFE(
            estimator=RandomForestClassifier(**RFE_BASE_ESTIMATOR_PARAMS),
            n_features_to_select=n,
            step=RFE_STEP,
            verbose=0,
        )
        rfe.fit(X_train, y_train)
        mask = rfe.support_
        idx = [j for j in range(n_cols) if mask[j]]
        X_train_sub = X_train[:, idx]
        X_val_sub = X_val[:, idx]
        clf = RandomForestClassifier(**RFE_BASE_ESTIMATOR_PARAMS)
        clf.fit(X_train_sub, y_train)
        y_val_pred = clf.predict(X_val_sub)
        f1 = f1_score(y_val, y_val_pred, zero_division=0.0)
        if verbose:
            _log(f"val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_features = [feature_list[j] for j in range(n_cols) if mask[j]]
    return best_features, best_f1


def _best_threshold_for_f1(y_true, y_proba, thresholds=None):
    """Return threshold in [thresholds] that maximizes F1. y_proba: P(class=1)."""
    if thresholds is None:
        thresholds = [t / 100.0 for t in range(15, 86, 5)]
    best_f1, best_t = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _fit_final_model(X_train, y_train, n_train, selected_features, training_feature_cols, verbose=False):
    """
    Fit final classifier: GradientBoosting for large segments, RandomForest otherwise.
    Uses class_weight/sample_weight for imbalance. Returns fitted model.
    """
    idx = [training_feature_cols.index(c) for c in selected_features]
    X_train_sub = X_train[:, idx]
    if n_train >= MIN_SAMPLES_FOR_GRADIENT_BOOSTING:
        if verbose:
            _log(f"    using GradientBoosting (n_train={n_train} >= {MIN_SAMPLES_FOR_GRADIENT_BOOSTING})")
        sample_weight = compute_sample_weight("balanced", y_train)
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_sub, y_train, sample_weight=sample_weight)
    else:
        if verbose:
            _log(f"    using RandomForest (n_train={n_train})")
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train_sub, y_train)
    return model, idx, X_train_sub


def train_router(router_id, train_df, val_df, lookups, min_samples=50, verbose=True):
    """
    Train all segments for this router. Routing dimensions are excluded from model input.
    Saves to EXPORTS_DIR / router_id. Returns list of dicts with segment, n_features, val_f1, status.
    """
    routing_dims, segments, fixed_features, fixed_per_segment = get_router_spec(router_id)
    if segments is None:
        raise ValueError(f"Unknown router_id: {router_id}")
    use_rfe = fixed_features is None and not fixed_per_segment
    if verbose and routing_dims:
        _log(f"  Routing dims (excluded from training): {routing_dims}")
        if fixed_per_segment:
            _log(f"  Per-segment fixed features (no RFE): {fixed_per_segment}")
        elif fixed_features:
            _log(f"  Fixed training features (no RFE): {fixed_features}")
        else:
            training_feature_cols_global = get_training_feature_cols(router_id)
            _log(f"  Training features ({len(training_feature_cols_global)}): used for model + RFE")

    out_base = EXPORTS_DIR / router_id
    out_base.mkdir(parents=True, exist_ok=True)
    lookups_dir = out_base / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)
    for col in CATEGORICAL_COLS:
        with open(lookups_dir / f"{col}.json", "w", encoding="utf-8") as f:
            json.dump(lookups.get(col, {}), f, sort_keys=True)
    with open(out_base / "routing_dimensions.json", "w", encoding="utf-8") as f:
        json.dump(routing_dims or [], f)

    results = []
    n_segments = len(segments)
    for seg_idx, seg_spec in enumerate(segments, 1):
        name = seg_spec[0]
        if verbose:
            _log(f"  Segment {seg_idx}/{n_segments}: {name} ...")
        mask_train = get_segment_mask(train_df, seg_spec, routing_dims or [])
        mask_val = get_segment_mask(val_df, seg_spec, routing_dims or [])
        sub_train = train_df.loc[mask_train]
        sub_val = val_df.loc[mask_val]
        n_train, n_val = len(sub_train), len(sub_val)
        n_pos = int((sub_train[TARGET_COL] == 1).sum())
        n_neg = n_train - n_pos

        if n_train < min_samples or n_pos == 0 or n_neg == 0:
            if verbose:
                _log(f"    skip (train n={n_train}, pos={n_pos})")
            results.append({
                "segment": name, "status": "SKIP", "n_train": n_train,
                "n_val": n_val, "val_f1": None, "n_features": None,
            })
            continue

        training_feature_cols = get_training_feature_cols(router_id, segment_name=name)
        if verbose:
            _log(f"    encoding train ({n_train} rows) + val ({n_val} rows) [training features only] ...")
        X_train, y_train = encode_df(sub_train, lookups, feature_list=training_feature_cols)
        X_val, y_val = encode_df(sub_val, lookups, feature_list=training_feature_cols)
        if use_rfe:
            if verbose:
                _log(f"    RFE (maximize val F1) ...")
            selected_features, _ = rfe_by_val_f1(
                X_train, y_train, X_val, y_val, feature_list=training_feature_cols, verbose=verbose
            )
            if verbose:
                _log(f"    training final model ({len(selected_features)} features) ...")
            model, idx, _ = _fit_final_model(
                X_train, y_train, n_train, selected_features, training_feature_cols, verbose=verbose
            )
            X_val_sub = X_val[:, idx]
        else:
            selected_features = list(training_feature_cols)
            idx = list(range(len(selected_features)))
            model, _, _ = _fit_final_model(
                X_train, y_train, n_train, selected_features, training_feature_cols, verbose=verbose
            )
            X_val_sub = X_val[:, idx] if idx else X_val

        # Threshold tuning: choose threshold that maximizes val F1 (instead of fixed 0.5)
        y_val_proba = model.predict_proba(X_val_sub)[:, 1]
        best_threshold = _best_threshold_for_f1(y_val, y_val_proba)
        val_f1 = float(f1_score(y_val, (y_val_proba >= best_threshold).astype(int), zero_division=0.0))

        seg_dir = out_base / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        model_dir = seg_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, model_dir / "model.pkl")
        with open(seg_dir / "selected_features.json", "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=0)
        with open(seg_dir / "threshold.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": best_threshold}, f)

        if verbose:
            _log(f"    done: val F1={val_f1:.4f}, threshold={best_threshold:.2f}, features={len(selected_features)}")
            _log(f"    selected: {selected_features}")
        results.append({
            "segment": name, "status": "OK", "n_train": n_train, "n_val": n_val,
            "val_f1": val_f1, "n_features": len(selected_features),
        })
    return results


def train_router_with_config(
    routing_dims,
    segments,
    train_df,
    val_df,
    lookups,
    out_dir,
    min_samples=50,
    verbose=False,
):
    """
    Train a router with explicit (routing_dims, segments); save to out_dir.
    Same logic as train_router but without router_id. Returns (results_list, training_feature_cols).
    """
    import joblib
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    lookups_dir = out_base / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)
    for col in CATEGORICAL_COLS:
        with open(lookups_dir / f"{col}.json", "w", encoding="utf-8") as f:
            json.dump(lookups.get(col, {}), f, sort_keys=True)
    with open(out_base / "routing_dimensions.json", "w", encoding="utf-8") as f:
        json.dump(routing_dims or [], f)
    with open(out_base / "segments.json", "w", encoding="utf-8") as f:
        json.dump([(name, fd) for name, fd in segments], f)

    training_feature_cols = get_training_feature_cols_for_dims(routing_dims)
    results = []
    n_seg = len(segments)
    for seg_idx, seg_spec in enumerate(segments):
        name = seg_spec[0]
        if not verbose and n_seg > 10 and ((seg_idx + 1) % 10 == 0 or seg_idx == 0):
            _log(f"      segment {seg_idx + 1}/{n_seg} ...")
        mask_train = get_segment_mask(train_df, seg_spec, routing_dims or [])
        mask_val = get_segment_mask(val_df, seg_spec, routing_dims or [])
        sub_train = train_df.loc[mask_train]
        sub_val = val_df.loc[mask_val]
        n_train, n_val = len(sub_train), len(sub_val)
        n_pos = int((sub_train[TARGET_COL] == 1).sum())
        n_neg = n_train - n_pos
        if n_train < min_samples or n_pos == 0 or n_neg == 0:
            results.append({
                "segment": name, "status": "SKIP", "n_train": n_train,
                "n_val": n_val, "val_f1": None, "n_features": None,
            })
            continue
        X_train, y_train = encode_df(sub_train, lookups, feature_list=training_feature_cols)
        X_val, y_val = encode_df(sub_val, lookups, feature_list=training_feature_cols)
        selected_features, _ = rfe_by_val_f1(
            X_train, y_train, X_val, y_val, feature_list=training_feature_cols, verbose=verbose
        )
        model, idx, _ = _fit_final_model(
            X_train, y_train, n_train, selected_features, training_feature_cols, verbose=verbose
        )
        X_val_sub = X_val[:, idx]
        y_val_proba = model.predict_proba(X_val_sub)[:, 1]
        best_threshold = _best_threshold_for_f1(y_val, y_val_proba)
        val_f1 = float(f1_score(y_val, (y_val_proba >= best_threshold).astype(int), zero_division=0.0))
        seg_dir = out_base / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        model_dir = seg_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.pkl")
        with open(seg_dir / "selected_features.json", "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=0)
        with open(seg_dir / "threshold.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": best_threshold}, f)
        results.append({
            "segment": name, "status": "OK", "n_train": n_train, "n_val": n_val,
            "val_f1": val_f1, "n_features": len(selected_features),
        })
    return results, training_feature_cols


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a single router (for use by run_router_optimization)")
    parser.add_argument("router_id", help="e.g. router_12, router_1")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_df, val_df, test_df, holdout_df = load_train_val_test_holdout()
    lookups = build_lookups_from_df(train_df)
    results = train_router(
        args.router_id, train_df, val_df, lookups,
        min_samples=args.min_samples, verbose=not args.quiet,
    )
    ok = [r for r in results if r["status"] == "OK"]
    print(f"Trained {len(ok)}/{len(results)} segments for {args.router_id}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
