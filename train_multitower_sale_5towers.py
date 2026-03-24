#!/usr/bin/env python3
"""
5-tower model: demo_age_gender, lead, housing, insurance, culture (with race) only.
Target: SALE_MADE_FLAG. Meta model (logistic regression) blends tower outputs.
Same structure and plotting as train_multitower_sale_only.py. Requires build_training_splits.py.
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

REPO_ROOT = Path(__file__).resolve().parent
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

# Demographics: gender + age (numeric)
FEATURES_DEMO_AGE_GENDER = [
    "GENDER", "AGE",
]
# Lead (contactability: Roku flag + phone linkage and activity)
FEATURES_LEAD_BASE = [
    "ROKU_FLAG",
    "PHONE_1_LINKAGE_SCORE", "PHONE_1_CONTACTABILITY_SCORE", "PHONE_1_USAGE_12_MONTH", "PHONE_1_USAGE_2_MONTH",
]
# Insurance (anything insurance or _ins_)
FEATURES_INSURANCE_BASE = [
    "LIFE_INS_LOYALTY_HIGH_PROPENSITY", "LIFE_INS_LOYALTY_LOW_PROPENSITY", "LIFE_INS_AFFINITY",
    "INSURANCE_EXPIRE_01_03", "INSURANCE_EXPIRE_03_06", "INSURANCE_EXPIRE_06_09", "INSURANCE_EXPIRE_09_12",
    "AUTO_INS_CLAIM_LEAST_LIKELY", "AUTO_INS_CLAIM_MOST_LIKELY", "PROPERTY_INS_CLAIM_LEAST_LIKELY", "PROPERTY_INS_CLAIM_MOST_LIKELY",
    "LIKELY_SHOPPING_AUTO_INS",
]
# Culture (language, ethnicity, religion, race)
FEATURES_CULTURE_BASE = [
    "LANGUAGE", "ETHNICITY_GROUP_V2", "RELIGION", "RACE_V2",
]
# Housing (own/rent, length of residence; home ownership tenure, property type/value/size)
# Optimized via backward elimination: DWELLING_TYPE removed (KS 0.2356 -> 0.2359)
FEATURES_HOUSING_BASE = [
    "HOMEOWNER_RENTER", "HOME_OWNER", "LENGTH_OF_RESIDENCE", "LENGTH_OF_RESIDENCE_BAND", "NEW_MOVER_FLAG",
    "HOME_OWNERSHIP_1_3_YEARS", "HOME_OWNERSHIP_3_7_YEARS", "HOME_OWNERSHIP_7_15_YEARS", "HOME_OWNERSHIP_LESS_THAN_1_YEAR", "HOME_OWNERSHIP_MORE_THAN_15_YEARS",
    "TIME_AT_CURRENT_ADDRESS_LONGEST", "TIME_AT_CURRENT_ADDRESS_SHORTEST",
    "DWELLING_UNIT_SIZE", "RENTER",
    "PROPERTY_TYPE_SINGLE_FAMILY", "PROPERTY_TYPE_MULTI_FAMILY_2_4", "PROPERTY_TYPE_MULTI_FAMILY_5_PLUS", "PROPERTY_TYPE_MANUFACTURED_HOUSE",
    "PROPERTY_VALUE_150", "PROPERTY_VALUE_250", "PROPERTY_VALUE_350", "PROPERTY_VALUE_500", "PROPERTY_VALUE_1000", "PROPERTY_VALUE_GREATER_THAN_1000",
    "PROPERTY_HOME_SIZE_1499", "PROPERTY_HOME_SIZE_2000", "PROPERTY_HOME_SIZE_3000", "PROPERTY_HOME_SIZE_3000_PLUS", "PROPERTY_HAS_POOL", "PROPERTY_HAS_GARAGE",
]

BUCKET_CATEGORICAL_COLS = [
    "AGE_5YR_BUCKET", "INCOME_10K_BUCKET",
    "BPS_TIER", "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET", "INCOME_BAND",
]

TIER_GOLD_MIN = 0.75
TIER_SILVER_MIN = 0.5
TIER_BRONZE_MIN = 0.25
USE_PRODUCT_META = False
EXPORT_DIR = EXPORTS_DIR / "multitower_sale_5towers"

AGE_5YR_EDGES = list(range(0, 86, 5)) + [999]
AGE_5YR_LABELS = [f"{a}-{a+4}" for a in range(0, 85, 5)] + ["85+"]
INCOME_10K_EDGES_THOUSANDS = list(range(0, 251, 10)) + [1e6]
INCOME_10K_LABELS = [f"{a}_{a+10}" for a in range(0, 250, 10)] + ["250_plus"]
INCOME_10K_EDGES_DOLLARS = list(range(0, 251_000, 10_000)) + [1e9]


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


def _tier_from_proba(proba, gold_min=0.75, silver_min=0.5, bronze_min=0.25):
    out = np.full(len(proba), "Tin", dtype=object)
    out[proba >= bronze_min] = "Bronze"
    out[proba >= silver_min] = "Silver"
    out[proba >= gold_min] = "Gold"
    return out


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


def main():
    print("Loading train/val/test ...")
    train_df, val_df, test_df, _ = load_train_val_test_holdout()
    train_df = _add_bucket_columns(train_df)
    val_df = _add_bucket_columns(val_df)
    test_df = _add_bucket_columns(test_df)
    for col in ["STABILITY_BUCKET", "BPS_TIER", "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"]:
        if col not in train_df.columns:
            train_df[col] = "unknown"
            val_df[col] = "unknown"
            test_df[col] = "unknown"
    lookups = _build_lookups_with_buckets(train_df)

    features_demo_age_gender = [c for c in FEATURES_DEMO_AGE_GENDER if c in train_df.columns]
    features_lead = [c for c in FEATURES_LEAD_BASE if c in train_df.columns]
    features_housing = [c for c in FEATURES_HOUSING_BASE if c in train_df.columns]
    # Insurance: base list only (no LOG_/INTERACT_/PROD_ engineered columns) to avoid KS loss from extra dims
    features_insurance = [c for c in FEATURES_INSURANCE_BASE if c in train_df.columns]
    features_culture = [c for c in FEATURES_CULTURE_BASE if c in train_df.columns]

    TOWER_NAMES = ["demo_age_gender", "lead", "housing", "insurance", "culture"]
    TOWER_FEATURES = [
        features_demo_age_gender, features_lead, features_housing, features_insurance, features_culture,
    ]

    target = "SALE_MADE_FLAG"
    y_train = train_df[target].astype(int).values

    print("\n--- 5-tower (demo_age_gender, lead, housing, insurance, culture) for SALE_MADE_FLAG ---")
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

    if USE_PRODUCT_META:
        meta = None
        def _product_proba(P_list):
            out = np.ones(len(P_list[0]), dtype=np.float64)
            for p in P_list:
                out *= np.asarray(p, dtype=np.float64)
            return out
    else:
        X_meta = np.column_stack(P_train_list)
        meta = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, class_weight="balanced", C=0.5)
        meta.fit(X_meta, y_train)

    P_test_list = []
    for name, t, feat_list in towers:
        X_t, _ = encode_df(test_df, lookups, feature_list=feat_list)
        P_test_list.append(t.predict_proba(X_t)[:, 1])
    if USE_PRODUCT_META:
        proba_sale = _product_proba(P_test_list)
    else:
        proba_sale = meta.predict_proba(np.column_stack(P_test_list))[:, 1]

    P_val_list = []
    for name, t, feat_list in towers:
        X_v, _ = encode_df(val_df, lookups, feature_list=feat_list)
        P_val_list.append(t.predict_proba(X_v)[:, 1])
    if USE_PRODUCT_META:
        proba_val = _product_proba(P_val_list)
    else:
        proba_val = meta.predict_proba(np.column_stack(P_val_list))[:, 1]
    y_val = val_df[target].astype(int).values
    best_f1, best_t = 0.0, 0.5
    if USE_PRODUCT_META:
        thresh_candidates = np.unique(np.clip(np.percentile(proba_val, np.linspace(50, 99, 20)), 1e-12, 1.0))
        thresh_candidates = np.concatenate([thresh_candidates, np.logspace(-10, -2, 30)])
        thresh_candidates = np.unique(np.clip(thresh_candidates, 1e-12, 1.0))
        for t in thresh_candidates:
            pred = (proba_val >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
    else:
        for t in [x / 100.0 for x in range(5, 76, 5)]:
            pred = (proba_val >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    pred_sale = (proba_sale >= best_t).astype(int)
    y_test = test_df[target].astype(int).values
    test_f1 = f1_score(y_test, pred_sale, zero_division=0.0)
    thresh_fmt = f"{best_t:.2e}" if best_t < 0.01 else f"{best_t:.2f}"
    print(f"  Meta: {'product of tower probas' if USE_PRODUCT_META else 'logistic regression'}")
    print(f"  Threshold (val F1): {thresh_fmt}  Test F1: {test_f1:.4f}")

    import joblib
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((towers, meta, USE_PRODUCT_META), EXPORT_DIR / "model.pkl")
    with open(EXPORT_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)
    lookups_dir = EXPORT_DIR / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)
    for col in list(CATEGORICAL_COLS) + BUCKET_CATEGORICAL_COLS:
        if col not in lookups:
            continue
        with open(lookups_dir / f"{col}.json", "w", encoding="utf-8") as f:
            json.dump(lookups.get(col, {}), f, sort_keys=True)

    tier = _tier_from_proba(proba_sale, gold_min=TIER_GOLD_MIN, silver_min=TIER_SILVER_MIN, bronze_min=TIER_BRONZE_MIN)
    out_df = pd.DataFrame(
        {f"proba_{n}": p for n, p in zip(TOWER_NAMES[: len(P_test_list)], P_test_list)},
    )
    out_df["proba_sale"] = proba_sale
    out_df["tier"] = tier
    out_df["pred_sale"] = pred_sale
    out_df["SALE_MADE_FLAG"] = y_test
    out_df.to_csv(REPO_ROOT / "multitower_sale_5towers_test.csv", index=False)
    print(f"  Wrote multitower_sale_5towers_test.csv ({len(out_df):,} rows)")

    # --- PNG: P(sale) distribution and combined tower probas by outcome ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sale_act = out_df["SALE_MADE_FLAG"].astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        p0 = out_df.loc[sale_act == 0, "proba_sale"].dropna()
        p1 = out_df.loc[sale_act == 1, "proba_sale"].dropna()
        if USE_PRODUCT_META and (len(p0) and len(p1)) and (p0.max() < 0.1 or p1.max() < 0.1):
            eps = max(1e-12, min(p0.min(), p1.min()) * 0.5)
            p0 = np.clip(p0, eps, None)
            p1 = np.clip(p1, eps, None)
            ax.hist(p0, bins=40, alpha=0.5, color="steelblue", density=True, label="No sale")
            ax.hist(p1, bins=40, alpha=0.5, color="coral", density=True, label="Sale")
            ax.axvline(best_t, color="black", linestyle="--", linewidth=1, label=f"Threshold={best_t:.2e}")
            ax.set_xscale("log")
            ax.set_xlabel("P(sale) (meta = product of towers, log scale)")
        else:
            ax.hist(p0, bins=30, alpha=0.5, color="steelblue", density=True, label="No sale")
            ax.hist(p1, bins=30, alpha=0.5, color="coral", density=True, label="Sale")
            thresh_lbl = f"{best_t:.2e}" if best_t < 0.01 else f"{best_t:.2f}"
            ax.axvline(best_t, color="black", linestyle="--", linewidth=1, label=f"Threshold={thresh_lbl}")
            ax.set_xlabel("P(sale) (meta output)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title("Predicted P(sale) distribution by SALE_MADE_FLAG (test)")
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        xcol = "proba_demo_age_gender"
        ycol = "proba_housing"
        ax.scatter(out_df.loc[sale_act == 0, xcol], out_df.loc[sale_act == 0, ycol], c="steelblue", alpha=0.35, s=10, label="No sale", edgecolors="none")
        ax.scatter(out_df.loc[sale_act == 1, xcol], out_df.loc[sale_act == 1, ycol], c="coral", alpha=0.6, s=14, label="Sale", edgecolors="none")
        ax.set_xlabel(f"P(sale) from {xcol.replace('proba_', '')} tower")
        ax.set_ylabel(f"P(sale) from {ycol.replace('proba_', '')} tower")
        ax.legend()
        ax.set_title("Combined: demo_age_gender vs housing tower probas by outcome (test)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        plt.suptitle("5-tower: P(sale) distribution and demo vs housing tower view", fontsize=11, y=1.02)
        plt.tight_layout()
        plt.savefig(REPO_ROOT / "multitower_sale_5towers_proba_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Wrote multitower_sale_5towers_proba_distributions.png")

        tower_cols = [c for c in out_df.columns if c.startswith("proba_") and c != "proba_sale"]
        n_towers = len(tower_cols)
        if n_towers >= 1:
            nrows = 2
            ncols = (n_towers + 1) // 2 if n_towers > 1 else 1
            if n_towers == 1:
                nrows, ncols = 1, 1
            elif n_towers <= 2:
                nrows, ncols = 1, n_towers
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
            if n_towers == 1:
                axes2 = np.array([axes2])
            else:
                axes2 = np.atleast_2d(axes2)
            for idx, col in enumerate(tower_cols):
                r, c = idx // ncols, idx % ncols
                ax = axes2[r, c]
                name = col.replace("proba_", "")
                ax.hist(out_df.loc[sale_act == 0, col].dropna(), bins=25, alpha=0.5, color="steelblue", density=True, label="No sale")
                ax.hist(out_df.loc[sale_act == 1, col].dropna(), bins=25, alpha=0.5, color="coral", density=True, label="Sale")
                ax.set_xlabel(f"P(sale) from {name} tower")
                ax.set_ylabel("Density")
                ax.legend(fontsize=8)
                ax.set_title(f"Tower: {name}")
                ax.grid(True, alpha=0.3)
            for idx in range(n_towers, nrows * ncols):
                r, c = idx // ncols, idx % ncols
                axes2[r, c].set_visible(False)
            plt.suptitle("5-tower: predicted proba distribution by tower (test)", fontsize=11, y=1.01)
            plt.tight_layout()
            plt.savefig(REPO_ROOT / "multitower_sale_5towers_proba_distributions_by_tower.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Wrote multitower_sale_5towers_proba_distributions_by_tower.png")
    except Exception as e:
        print(f"  Proba distributions PNG skip: {e}")

    tier_order = ["Gold", "Silver", "Bronze", "Tin"]
    tier_rows = []
    tier_confusion = []
    tier_cm_per = {}
    for t in tier_order:
        mask = tier == t
        n = mask.sum()
        n_no_sale = int((out_df.loc[mask, "SALE_MADE_FLAG"] == 0).sum())
        n_sale = int((out_df.loc[mask, "SALE_MADE_FLAG"] == 1).sum())
        sale_rate = (n_sale / n) if n > 0 else 0.0
        tier_rows.append({"tier": t, "n": int(n), "n_no_sale": n_no_sale, "n_sale": n_sale, "sale_rate": round(sale_rate, 4)})
        tier_confusion.append((t, n_no_sale, n_sale))
        sub = out_df.loc[mask]
        if len(sub) > 0:
            y_t = sub["SALE_MADE_FLAG"].astype(int).values
            y_p = sub["pred_sale"].astype(int).values
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            tn = int(((y_t == 0) & (y_p == 0)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            tier_cm_per[t] = (tn, fp, fn, tp)
        else:
            tier_cm_per[t] = (0, 0, 0, 0)

    print("\n--- Tier analysis (Gold>=0.75, Silver>=0.5, Bronze>=0.25, Tin<0.25) ---")
    for r in tier_rows:
        print(f"  {r['tier']:8s}  n={r['n']:5d}  no_sale={r['n_no_sale']:5d}  sale={r['n_sale']:5d}  sale_rate={r['sale_rate']:.4f}")

    proba = out_df["proba_sale"].values
    y = out_df["SALE_MADE_FLAG"].astype(int).values
    p_sale = proba[y == 1]
    p_no_sale = proba[y == 0]
    ks_stat, ks_pvalue = ks_2samp(p_sale, p_no_sale)
    print(f"\n  KS (test): {ks_stat:.4f}  (P(sale) | sale=1 vs sale=0)")

    pd.DataFrame(tier_rows).to_csv(REPO_ROOT / "multitower_sale_5towers_tier_analysis.csv", index=False)
    with pd.ExcelWriter(REPO_ROOT / "multitower_sale_5towers_tier_analysis.xlsx", engine="openpyxl") as xl:
        pd.DataFrame(tier_rows).to_excel(xl, sheet_name="Tier_summary", index=False)
        pd.DataFrame([(t, a, b) for t, a, b in tier_confusion], columns=["tier", "No_sale", "Sale"]).to_excel(xl, sheet_name="Tier_x_Outcome", index=False)
        for t in tier_order:
            tn, fp, fn, tp = tier_cm_per[t]
            pd.DataFrame([["Tier", t], ["Actual 0", "", tn, fp], ["Actual 1", "", fn, tp]]).to_excel(xl, sheet_name=f"CM_{t}"[:31], index=False, header=False)
    print(f"  Wrote multitower_sale_5towers_tier_analysis.csv and .xlsx")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(tier_order))
        w = 0.35
        ax1.bar(x - w / 2, [r["n_no_sale"] for r in tier_rows], width=w, label="No sale", color="steelblue")
        ax1.bar(x + w / 2, [r["n_sale"] for r in tier_rows], width=w, label="Sale", color="coral")
        ax1.set_xticks(x)
        ax1.set_xticklabels(tier_order)
        ax1.set_ylabel("Count (test)")
        ax1.set_xlabel("Tier")
        ax1.legend()
        ax1.set_title("5-tower: sales vs non-sales per tier (Gold>=0.75, Silver>=0.5, Bronze>=0.25, Tin<0.25)")
        ax1.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(REPO_ROOT / "multitower_sale_5towers_tier_counts.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Wrote multitower_sale_5towers_tier_counts.png")

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        mat = np.array([[a, b] for _, a, b in tier_confusion], dtype=float)
        im = ax2.imshow(mat, cmap="Blues", aspect="auto")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["No sale", "Sale"])
        ax2.set_yticks(range(len(tier_order)))
        ax2.set_yticklabels(tier_order)
        for i in range(len(tier_order)):
            for j in range(2):
                ax2.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=12)
        plt.colorbar(im, ax=ax2, label="Count")
        plt.tight_layout()
        plt.savefig(REPO_ROOT / "multitower_sale_5towers_tier_confusion_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Wrote multitower_sale_5towers_tier_confusion_heatmap.png")

        fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
        axes3 = axes3.flatten()
        for idx, t in enumerate(tier_order):
            ax = axes3[idx]
            tn, fp, fn, tp = tier_cm_per[t]
            mat = np.array([[tn, fp], [fn, tp]], dtype=float)
            ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["Act 0", "Act 1"])
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, int(mat[ii, jj]), ha="center", va="center", fontsize=11)
            ax.set_title(f"{t} (n={tier_rows[idx]['n']})")
        plt.suptitle("5-tower: per-tier confusion (Pred sale vs Actual sale)")
        plt.tight_layout()
        plt.savefig(REPO_ROOT / "multitower_sale_5towers_tier_confusion_matrices.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Wrote multitower_sale_5towers_tier_confusion_matrices.png")
    except Exception as e:
        print(f"  Tier PNGs skip: {e}")

    analysis_df = pd.DataFrame({"PREDICTION_PROBA": proba_sale, "PREDICTION": pred_sale, "SALE_MADE_FLAG": y_test})
    df = analysis_df.copy()
    proba = pd.to_numeric(df["PREDICTION_PROBA"], errors="coerce")
    valid = proba.notna()
    df = df.loc[valid].copy()
    proba = proba.loc[valid]
    if len(df) < 10:
        print("  Too few rows for decile analysis")
        return 0
    try:
        df["decile"] = pd.qcut(proba, q=10, labels=False, duplicates="drop") + 1
    except Exception:
        df["decile"] = pd.qcut(proba, q=min(10, max(2, int(proba.nunique()))), labels=False, duplicates="drop") + 1

    def confusion_stats(yt, yp):
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        n = len(yt)
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return (tn, fp, fn, tp), acc, prec, rec, spec, f1

    rows = []
    for d in sorted(df["decile"].dropna().unique()):
        sub = df.loc[df["decile"] == d]
        y_t = sub["SALE_MADE_FLAG"].astype(int).values
        y_p = sub["PREDICTION"].astype(int).values
        (tn, fp, fn, tp), acc, prec, rec, spec, f1 = confusion_stats(y_t, y_p)
        proba_sub = sub["PREDICTION_PROBA"].astype(float)
        actual_rate = y_t.mean() if len(y_t) else 0.0
        rows.append({
            "decile": int(d), "n": len(sub),
            "proba_min": round(proba_sub.min(), 4), "proba_max": round(proba_sub.max(), 4), "proba_mean": round(proba_sub.mean(), 4),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "specificity": round(spec, 4), "f1": round(f1, 4),
            "actual_sale_rate": round(actual_rate, 4),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(REPO_ROOT / "multitower_sale_5towers_decile_analysis.csv", index=False)
    print(f"  Wrote multitower_sale_5towers_decile_analysis.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deciles = stats_df["decile"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(deciles))
        w = 0.25
        ax.bar(x - w, stats_df["f1"], width=w, label="F1", color="steelblue")
        ax.bar(x, stats_df["precision"], width=w, label="Precision", color="coral")
        ax.bar(x + w, stats_df["recall"], width=w, label="Recall", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(deciles)
        ax.set_xlabel("Decile of P(sale) (1=lowest, 10=highest)")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.set_title("5-tower decile analysis: SALE_MADE_FLAG by P(sale)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPO_ROOT / "multitower_sale_5towers_decile_analysis.png", dpi=150)
        plt.close()
        print(f"  Wrote multitower_sale_5towers_decile_analysis.png")
    except Exception as e:
        print(f"  Decile PNG skip: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
