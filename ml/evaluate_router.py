"""
Evaluate a trained router on a dataset (val or test): route each row to segment model, predict, aggregate F1.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from .config import (
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    EXPORTS_DIR,
)
from .routers import (
    get_all_routers,
    get_segment_name_for_row,
    get_segment_name_for_row_given_spec,
)


def load_router_models_and_features(router_id):
    """
    Load models and selected_features for each segment of this router.
    Returns (models dict segment_name -> model, feature_lists dict segment_name -> list).
    """
    import joblib
    routers = {rid: segs for rid, _rdim, segs, *_ in get_all_routers()}
    if router_id not in routers:
        raise ValueError(f"Unknown router_id: {router_id}")
    out_base = EXPORTS_DIR / router_id
    if not out_base.is_dir():
        raise FileNotFoundError(f"Router dir not found: {out_base}. Train first.")
    models = {}
    feature_lists = {}
    for seg_spec in routers[router_id]:
        name = seg_spec[0]
        model_dir = out_base / name / "model"
        pkl_path = model_dir / "model.pkl"
        if not pkl_path.exists():
            for p in (out_base / name).rglob("model.pkl"):
                pkl_path = p
                break
        if not pkl_path.exists():
            continue
        models[name] = joblib.load(pkl_path)
        sel_path = out_base / name / "selected_features.json"
        if sel_path.exists():
            with open(sel_path, "r", encoding="utf-8") as f:
                feature_lists[name] = json.load(f)
        else:
            from .routers import get_training_feature_cols
            feature_lists[name] = get_training_feature_cols(router_id, segment_name=name)
    return models, feature_lists


def load_lookups(router_id):
    """Load label encoding lookups from router's lookups dir."""
    lookups_dir = EXPORTS_DIR / router_id / "lookups"
    lookups = {}
    for col in CATEGORICAL_COLS:
        path = lookups_dir / f"{col}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lookups[col] = json.load(f)
        else:
            lookups[col] = {}
    return lookups


def load_thresholds(router_id):
    """Load per-segment decision thresholds (from threshold tuning). Returns dict segment_name -> float (default 0.5 if missing)."""
    routers = {rid: segs for rid, _rdim, segs, *_ in get_all_routers()}
    if router_id not in routers:
        return {}
    out_base = EXPORTS_DIR / router_id
    thresholds = {}
    for seg_spec in routers[router_id]:
        name = seg_spec[0]
        path = out_base / name / "threshold.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                thresholds[name] = float(json.load(f).get("threshold", 0.5))
        else:
            thresholds[name] = 0.5
    return thresholds


def encode_row(row, feature_list, lookups):
    """Encode one row to float list in feature_list order."""
    vec = []
    for col in feature_list:
        raw = row.get(col, "")
        if col in NUMERIC_COLS:
            try:
                vec.append(float(raw) if raw not in (None, "", "nan") else 0.0)
            except (TypeError, ValueError):
                vec.append(0.0)
        else:
            key = str(raw).strip() if raw not in (None, "") else ""
            vec.append(float(lookups.get(col, {}).get(key, 0)))
    return vec


def load_router_from_dir(dir_path):
    """
    Load a router saved by train_router_with_config from dir_path.
    Returns (routing_dims, segments, models, feature_lists, lookups).
    """
    import joblib
    dir_path = Path(dir_path)
    with open(dir_path / "routing_dimensions.json", "r", encoding="utf-8") as f:
        routing_dims = json.load(f)
    with open(dir_path / "segments.json", "r", encoding="utf-8") as f:
        segments = [tuple(x) for x in json.load(f)]
    lookups = {}
    for col in CATEGORICAL_COLS:
        p = dir_path / "lookups" / f"{col}.json"
        lookups[col] = json.load(p.open("r", encoding="utf-8")) if p.exists() else {}
    models = {}
    feature_lists = {}
    thresholds = {}
    for name, _ in segments:
        pkl = dir_path / name / "model" / "model.pkl"
        if not pkl.exists():
            for p in (dir_path / name).rglob("model.pkl"):
                pkl = p
                break
        if not pkl.exists():
            continue
        models[name] = joblib.load(pkl)
        sel = dir_path / name / "selected_features.json"
        feature_lists[name] = json.load(sel.open("r", encoding="utf-8")) if sel.exists() else list(FEATURE_COLS)
        th = dir_path / name / "threshold.json"
        thresholds[name] = float(json.load(th.open("r", encoding="utf-8")).get("threshold", 0.5)) if th.exists() else 0.5
    return routing_dims, segments, models, feature_lists, lookups, thresholds


def evaluate_router_from_dir(dir_path, df):
    """
    Evaluate a router loaded from dir_path (saved by train_router_with_config) on df.
    Returns same metrics dict as evaluate_router_on_df.
    """
    routing_dims, segments, models, feature_lists, lookups, thresholds = load_router_from_dir(dir_path)
    if not models:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n_scored": 0}

    y_true = df[TARGET_COL].astype(int).values
    y_pred = np.full(len(df), -1, dtype=int)
    y_proba = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        row = df.iloc[i]
        seg_name = get_segment_name_for_row_given_spec(routing_dims, segments, row)
        if seg_name is None or seg_name not in models:
            continue
        thresh = thresholds.get(seg_name, 0.5)
        model = models[seg_name]
        fl = feature_lists.get(seg_name, FEATURE_COLS)
        vec = encode_row(row, fl, lookups)
        X = np.asarray([vec], dtype=np.float64)
        try:
            p = model.predict_proba(X)[0, 1]
            y_proba[i] = p
            y_pred[i] = 1 if p >= thresh else 0
        except Exception:
            pass
    valid = y_pred >= 0
    n_scored = int(valid.sum())
    if n_scored == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n_scored": 0, "n_total": len(df)}
    y_t = y_true[valid]
    y_p = y_pred[valid].astype(int)
    return {
        "f1": float(f1_score(y_t, y_p, zero_division=0.0)),
        "precision": float(precision_score(y_t, y_p, zero_division=0.0)),
        "recall": float(recall_score(y_t, y_p, zero_division=0.0)),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "n_scored": n_scored,
        "n_total": len(df),
    }


def predict_router_on_df(router_id, df, models=None, feature_lists=None, lookups=None, thresholds=None):
    """
    Route each row to segment model and predict in batch per segment.
    Uses per-segment threshold if available (from threshold tuning); else 0.5.
    Returns (y_pred, y_proba, segment_names) where segment_names[i] is the segment used for row i (or "" if unscored).
    """
    if models is None or feature_lists is None or lookups is None:
        models, feature_lists = load_router_models_and_features(router_id)
        lookups = load_lookups(router_id)
    if thresholds is None:
        thresholds = load_thresholds(router_id)
    n = len(df)
    y_pred = np.full(n, -1, dtype=int)
    y_proba = np.full(n, np.nan, dtype=float)
    segment_names = [""] * n

    if not models:
        return y_pred, y_proba, segment_names

    # One pass: segment per row
    seg_per_row = [get_segment_name_for_row(router_id, df.iloc[i]) for i in range(n)]
    seg_to_indices = defaultdict(list)
    for i, seg in enumerate(seg_per_row):
        if seg and seg in models:
            seg_to_indices[seg].append(i)

    # Batch predict per segment (use tuned threshold per segment)
    for seg_name, indices in seg_to_indices.items():
        thresh = thresholds.get(seg_name, 0.5)
        model = models[seg_name]
        fl = feature_lists.get(seg_name, FEATURE_COLS)
        X = np.array([encode_row(df.iloc[i], fl, lookups) for i in indices], dtype=np.float64)
        try:
            proba = model.predict_proba(X)[:, 1]
            for k, idx in enumerate(indices):
                y_proba[idx] = proba[k]
                y_pred[idx] = 1 if proba[k] >= thresh else 0
                segment_names[idx] = seg_name
        except Exception:
            pass

    return y_pred, y_proba, segment_names


def evaluate_router_on_df(router_id, df, models=None, feature_lists=None, lookups=None):
    """
    Route each row to the correct segment model, predict, and compute metrics.
    Returns dict with f1, precision, recall, accuracy, and per-segment counts.
    If models/feature_lists/lookups are None, they are loaded from EXPORTS_DIR/router_id.
    """
    y_pred, _, _ = predict_router_on_df(router_id, df, models, feature_lists, lookups)
    y_true = df[TARGET_COL].astype(int).values
    valid = y_pred >= 0
    n_scored = int(valid.sum())
    if n_scored == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n_scored": 0, "n_total": len(df)}

    y_t = y_true[valid]
    y_p = y_pred[valid].astype(int)
    return {
        "f1": float(f1_score(y_t, y_p, zero_division=0.0)),
        "precision": float(precision_score(y_t, y_p, zero_division=0.0)),
        "recall": float(recall_score(y_t, y_p, zero_division=0.0)),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "n_scored": n_scored,
        "n_total": len(df),
    }
