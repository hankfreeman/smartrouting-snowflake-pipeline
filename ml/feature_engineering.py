"""
Rigorous feature engineering: hundreds of new features via
log transforms, products, binary AND/OR, counts, thresholds, categorical crosses.
Call add_engineered_features(df) to add all; returns (df, ENGINEERED_NUMERIC_COLS, ENGINEERED_CATEGORICAL_COLS).
"""
import numpy as np
import pandas as pd


def _safe_log1p(series):
    return np.log1p(pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0))


def _safe_numeric(series, fill=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def _safe_binary(series):
    x = pd.to_numeric(series, errors="coerce").fillna(0)
    return (x > 0).astype(int)


# Columns to count for data-availability (non-null = data point available)
DATA_AVAILABILITY_COLS = [
    "AGE", "GENDER", "EST_HH_INCOME", "URBANICITY", "STATE", "EDUCATION", "RACE_V2",
    "MARITAL_STATUS", "LENGTH_OF_RESIDENCE", "BPS", "BPS_ABILITY_LETTER",
    "CREDIT_BEHAVIOR_ESTABLISHED", "CREDIT_BEHAVIOR_THRIVING", "CREDIT_BEHAVIOR_UP_AND_COMING",
    "AGG_CREDIT_TIER_1ST", "AGG_CREDIT_TIER_2ND", "AGG_CREDIT_TIER_3RD", "AGG_CREDIT_TIER_4TH",
    "CARD_SPENDERS_HIGH", "CARD_SPENDERS_LOW", "OPEN_PERSONAL_LOAN", "OPEN_STUDENT_LOAN",
    "HOMEOWNER_RENTER", "NUMBER_OF_CHILDREN_V2", "OCCUPATION_GROUP", "MSA",
]


def add_engineered_features(df):
    """
    Add hundreds of engineered features. Modifies df in place and returns
    (df, new_numeric_cols, new_categorical_cols).
    """
    out_numeric = []
    out_categorical = []

    # ---- 0. Data availability: count of non-null (available) data points ----
    cols_to_count = [c for c in DATA_AVAILABILITY_COLS if c in df.columns]
    if cols_to_count:
        # Non-null: for numeric, notna(); for object, notna() and str strip not empty
        def _non_null(s):
            if s.dtype.kind in "fcbiu":
                return s.notna().astype(int)
            return (s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.upper() != "NAN")).astype(int)
        count = pd.concat([_non_null(df[c]) for c in cols_to_count], axis=1).sum(axis=1)
        df["COUNT_NON_NULL"] = count
        df["DATA_AVAILABILITY_RATIO"] = count / len(cols_to_count)
        out_numeric.extend(["COUNT_NON_NULL", "DATA_AVAILABILITY_RATIO"])

    # ---- 1. Log transforms (log1p) for numerics ----
    log_candidates = [
        "LENGTH_OF_RESIDENCE", "NEW_MOVER", "CREDIT_BEHAVIOR_ESTABLISHED", "CREDIT_BEHAVIOR_THRIVING",
        "CREDIT_BEHAVIOR_UP_AND_COMING", "CARD_SPENDERS_HIGH", "CARD_SPENDERS_LOW",
        "OPEN_PERSONAL_LOAN", "OPEN_STUDENT_LOAN", "VETERAN_IN_HOUSEHOLD", "LIKELY_TO_MOVE",
        "LIFE_INS_LOYALTY_HIGH_PROPENSITY", "LIFE_INS_LOYALTY_LOW_PROPENSITY",
        "AGG_CREDIT_TIER_1ST", "AGG_CREDIT_TIER_2ND", "AGG_CREDIT_TIER_3RD", "AGG_CREDIT_TIER_4TH",
        "AGE", "EST_HH_INCOME", "CREDIT_BEHAVIOR_SUM", "CREDIT_TIER_COUNT",
    ]
    for col in log_candidates:
        if col not in df.columns:
            continue
        name = f"LOG_{col}"
        df[name] = _safe_log1p(df[col])
        out_numeric.append(name)

    # ---- 2. Products (interactions) ----
    # Pairs (col1, col2): product of normalized (0-1 scale) or raw
    interact_pairs = [
        ("AGE", "EST_HH_INCOME"), ("AGE", "CREDIT_BEHAVIOR_SUM"), ("AGE", "INCOME_LOG"),
        ("INCOME_LOG", "CREDIT_BEHAVIOR_SUM"), ("INCOME_LOG", "CREDIT_TIER_COUNT"),
        ("LENGTH_OF_RESIDENCE", "AGE"), ("CREDIT_BEHAVIOR_SUM", "CARD_SPENDER_ANY"),
        ("ROKU_FLAG", "AGE_70_PLUS"), ("ROKU_FLAG", "HIGH_INCOME_FLAG"), ("ROKU_FLAG", "INCOME_LOG"),
        ("BPS_PRESENT_FLAG", "HIGH_INCOME_FLAG"), ("BPS_PRESENT_FLAG", "INCOME_LOG"),
        ("AGE_70_PLUS", "HIGH_INCOME_FLAG"), ("AGE_70_PLUS", "INCOME_LOG"),
        ("NEW_MOVER_FLAG", "LIKELY_TO_MOVE_FLAG"), ("NEW_MOVER_FLAG", "LIFE_INS_AFFINITY"),
        ("CARD_SPENDER_ANY", "HAS_ANY_LOAN"), ("LIFE_INS_AFFINITY", "AGE_70_PLUS"),
        ("LIFE_EVENT_FLAG", "LIFE_INS_AFFINITY"), ("LIFE_EVENT_FLAG", "HIGH_INCOME_FLAG"),
        ("VETERAN_FLAG", "AGE_70_PLUS"), ("SENIOR_AFFLUENT", "BPS_PRESENT_FLAG"),
        ("LOG_AGE", "LOG_EST_HH_INCOME"), ("LOG_CREDIT_BEHAVIOR_SUM", "LOG_EST_HH_INCOME"),
    ]
    for c1, c2 in interact_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        name = f"INTERACT_{c1}_x_{c2}"
        v1 = _safe_numeric(df[c1])
        v2 = _safe_numeric(df[c2])
        df[name] = v1 * v2
        out_numeric.append(name)

    # ---- 3. Binary AND (both flags present) ----
    binary_cols = [
        "HIGH_INCOME_FLAG", "AGE_70_PLUS", "ROKU_FLAG", "BPS_PRESENT_FLAG", "NEW_MOVER_FLAG",
        "VETERAN_FLAG", "LIKELY_TO_MOVE_FLAG", "CARD_SPENDER_ANY", "HAS_ANY_LOAN",
        "SENIOR_AFFLUENT", "LIFE_EVENT_FLAG",
    ]
    binary_cols = [c for c in binary_cols if c in df.columns]
    and_pairs = [
        ("AGE_70_PLUS", "HIGH_INCOME_FLAG"), ("ROKU_FLAG", "BPS_PRESENT_FLAG"), ("ROKU_FLAG", "HIGH_INCOME_FLAG"),
        ("BPS_PRESENT_FLAG", "CARD_SPENDER_ANY"), ("BPS_PRESENT_FLAG", "HIGH_INCOME_FLAG"),
        ("CARD_SPENDER_ANY", "HAS_ANY_LOAN"), ("NEW_MOVER_FLAG", "LIKELY_TO_MOVE_FLAG"),
        ("AGE_70_PLUS", "BPS_PRESENT_FLAG"), ("AGE_70_PLUS", "CARD_SPENDER_ANY"),
        ("VETERAN_FLAG", "AGE_70_PLUS"), ("LIFE_EVENT_FLAG", "HIGH_INCOME_FLAG"),
        ("SENIOR_AFFLUENT", "BPS_PRESENT_FLAG"), ("SENIOR_AFFLUENT", "CARD_SPENDER_ANY"),
        ("ROKU_FLAG", "CARD_SPENDER_ANY"), ("ROKU_FLAG", "AGE_70_PLUS"),
    ]
    for c1, c2 in and_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        name = f"AND_{c1}_{c2}"
        df[name] = _safe_binary(df[c1]) * _safe_binary(df[c2])
        out_numeric.append(name)
    # All pairs of binary_cols (up to 55 pairs for 11 cols)
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i + 1 :]:
            name = f"AND_{c1}_{c2}"
            if name in df.columns:
                continue
            df[name] = _safe_binary(df[c1]) * _safe_binary(df[c2])
            out_numeric.append(name)

    # ---- 4. Binary OR (either flag present) ----
    or_pairs = [
        ("NEW_MOVER_FLAG", "LIKELY_TO_MOVE_FLAG"), ("CARD_SPENDER_ANY", "HAS_ANY_LOAN"),
        ("AGE_70_PLUS", "VETERAN_FLAG"), ("ROKU_FLAG", "BPS_PRESENT_FLAG"),
        ("HIGH_INCOME_FLAG", "CARD_SPENDER_ANY"), ("BPS_PRESENT_FLAG", "HIGH_INCOME_FLAG"),
    ]
    for c1, c2 in or_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        name = f"OR_{c1}_{c2}"
        df[name] = ((_safe_binary(df[c1]) + _safe_binary(df[c2])) >= 1).astype(int)
        out_numeric.append(name)
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i + 1 :]:
            name = f"OR_{c1}_{c2}"
            if name in df.columns:
                continue
            df[name] = ((_safe_binary(df[c1]) + _safe_binary(df[c2])) >= 1).astype(int)
            out_numeric.append(name)

    # ---- 5. Count features (sum of binaries) ----
    count_groups = [
        ("COUNT_MOBILITY", ["NEW_MOVER_FLAG", "LIKELY_TO_MOVE_FLAG"]),
        ("COUNT_FINANCIAL", ["CARD_SPENDER_ANY", "HAS_ANY_LOAN", "HIGH_INCOME_FLAG"]),
        ("COUNT_LIFE", ["AGE_70_PLUS", "LIFE_EVENT_FLAG", "VETERAN_FLAG"]),
        ("COUNT_ENGAGEMENT", ["ROKU_FLAG", "BPS_PRESENT_FLAG", "CARD_SPENDER_ANY", "HAS_ANY_LOAN"]),
        ("COUNT_SENIOR_SIGNALS", ["AGE_70_PLUS", "VETERAN_FLAG", "SENIOR_AFFLUENT"]),
        ("COUNT_ALL_FLAGS", binary_cols),
    ]
    for name, cols in count_groups:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        df[name] = sum(_safe_binary(df[c]) for c in cols)
        out_numeric.append(name)

    # ---- 6. Threshold binaries ----
    thresh_rules = [
        ("THRESH_AGE_55_PLUS", "AGE", 55),
        ("THRESH_AGE_60_PLUS", "AGE", 60),
        ("THRESH_AGE_65_PLUS", "AGE", 65),
        ("THRESH_AGE_75_PLUS", "AGE", 75),
        ("THRESH_INCOME_50K_PLUS", "EST_HH_INCOME", 50),
        ("THRESH_INCOME_75K_PLUS", "EST_HH_INCOME", 75),
        ("THRESH_INCOME_100K_PLUS", "EST_HH_INCOME", 100),
        ("THRESH_INCOME_150K_PLUS", "EST_HH_INCOME", 150),
        ("THRESH_CREDIT_BEHAVIOR_HIGH", "CREDIT_BEHAVIOR_SUM", 1.5),
        ("THRESH_LOR_5Y_PLUS", "LENGTH_OF_RESIDENCE", 5),
        ("THRESH_LOR_10_PLUS", "LENGTH_OF_RESIDENCE", 10),
        ("THRESH_CREDIT_TIERS_2_PLUS", "CREDIT_TIER_COUNT", 2),
        ("THRESH_CREDIT_TIERS_3_PLUS", "CREDIT_TIER_COUNT", 3),
        ("THRESH_LIFE_AFFINITY_POS", "LIFE_INS_AFFINITY", 0),
    ]
    for name, col, thresh in thresh_rules:
        if col not in df.columns:
            continue
        v = _safe_numeric(df[col])
        if "INCOME" in col and v.max() < 2000:
            v = v * 1000  # assume thousands
        df[name] = (v >= thresh).astype(int)
        out_numeric.append(name)

    # ---- 7. More products: numeric * binary ----
    num_x_binary = [
        ("INCOME_LOG", "ROKU_FLAG"), ("INCOME_LOG", "BPS_PRESENT_FLAG"), ("AGE", "HIGH_INCOME_FLAG"),
        ("CREDIT_BEHAVIOR_SUM", "BPS_PRESENT_FLAG"), ("LENGTH_OF_RESIDENCE", "NEW_MOVER_FLAG"),
        ("AGE", "CARD_SPENDER_ANY"), ("EST_HH_INCOME", "AGE_70_PLUS"),
        ("CREDIT_TIER_COUNT", "CARD_SPENDER_ANY"), ("LIFE_INS_AFFINITY", "AGE_70_PLUS"),
    ]
    for num_col, bin_col in num_x_binary:
        if num_col not in df.columns or bin_col not in df.columns:
            continue
        name = f"PROD_{num_col}_x_{bin_col}"
        df[name] = _safe_numeric(df[num_col]) * _safe_binary(df[bin_col])
        out_numeric.append(name)

    # ---- 8. Categorical crosses ----
    cross_pairs = [
        ("BPS_TIER", "STABILITY_BUCKET"), ("BPS_TIER", "CREDIT_QUALITY_BUCKET"),
        ("STABILITY_BUCKET", "CREDIT_QUALITY_BUCKET"), ("LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"),
        ("BPS_TIER", "FINANCIAL_ENGAGEMENT_BUCKET"), ("STABILITY_BUCKET", "LIFE_READINESS_BUCKET"),
        ("CREDIT_QUALITY_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"), ("INCOME_BAND", "AGE_SEG"),
        ("BPS_TIER", "LIFE_READINESS_BUCKET"), ("GENDER", "AGE_SEG"),
    ]
    for c1, c2 in cross_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        name = f"CROSS_{c1}_{c2}"
        s1 = df[c1].astype(str).str.strip().replace("nan", "")
        s2 = df[c2].astype(str).str.strip().replace("nan", "")
        df[name] = (s1 + "_" + s2).replace("^_|_$", "", regex=True).replace("^$", "unknown")
        out_categorical.append(name)

    # ---- 9. Ratio-style features ----
    ratio_pairs = [
        ("RATIO_INCOME_CREDIT", "EST_HH_INCOME", "CREDIT_BEHAVIOR_SUM"),
        ("RATIO_AGE_LOR", "AGE", "LENGTH_OF_RESIDENCE"),
    ]
    for name, num_col, den_col in ratio_pairs:
        if num_col not in df.columns or den_col not in df.columns:
            continue
        num = _safe_numeric(df[num_col])
        den = _safe_numeric(df[den_col]).replace(0, np.nan)
        if "INCOME" in num_col and num.max() < 2000:
            num = num * 1000
        df[name] = (num / den).fillna(0).clip(-1e6, 1e6)
        out_numeric.append(name)

    # ---- 10. Squared terms (nonlinear) ----
    sq_cols = ["AGE", "INCOME_LOG", "CREDIT_BEHAVIOR_SUM", "LENGTH_OF_RESIDENCE"]
    for col in sq_cols:
        if col not in df.columns:
            continue
        name = f"SQ_{col}"
        v = _safe_numeric(df[col])
        df[name] = v * v
        out_numeric.append(name)

    # ---- 11. Triple AND ----
    triple_and = [
        ("AND3_SENIOR_AFFLUENT_BPS", ["AGE_70_PLUS", "HIGH_INCOME_FLAG", "BPS_PRESENT_FLAG"]),
        ("AND3_ROKU_HIGH_CARD", ["ROKU_FLAG", "HIGH_INCOME_FLAG", "CARD_SPENDER_ANY"]),
        ("AND3_MOBILE_LIFE_EVENT", ["NEW_MOVER_FLAG", "LIKELY_TO_MOVE_FLAG", "LIFE_EVENT_FLAG"]),
        ("AND3_CREDIT_ENGAGED", ["CARD_SPENDER_ANY", "HAS_ANY_LOAN", "HIGH_INCOME_FLAG"]),
    ]
    for name, cols in triple_and:
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            continue
        prod = _safe_binary(df[cols[0]])
        for c in cols[1:]:
            prod = prod * _safe_binary(df[c])
        df[name] = prod
        out_numeric.append(name)

    # Dedupe
    out_numeric = list(dict.fromkeys(out_numeric))
    out_categorical = list(dict.fromkeys(out_categorical))

    return df, out_numeric, out_categorical


def get_all_engineered_names():
    """
    Return (engineered_numeric_names, engineered_categorical_names) that can be produced.
    Uses a minimal df with all expected base columns so the exact name list is deterministic.
    """
    base_cols = [
        "AGE", "EST_HH_INCOME", "INCOME_LOG", "CREDIT_BEHAVIOR_SUM", "CREDIT_TIER_COUNT",
        "LENGTH_OF_RESIDENCE", "NEW_MOVER", "CREDIT_BEHAVIOR_ESTABLISHED", "CREDIT_BEHAVIOR_THRIVING",
        "CREDIT_BEHAVIOR_UP_AND_COMING", "CARD_SPENDERS_HIGH", "CARD_SPENDERS_LOW",
        "OPEN_PERSONAL_LOAN", "OPEN_STUDENT_LOAN", "VETERAN_IN_HOUSEHOLD", "LIKELY_TO_MOVE",
        "LIFE_INS_LOYALTY_HIGH_PROPENSITY", "LIFE_INS_LOYALTY_LOW_PROPENSITY",
        "AGG_CREDIT_TIER_1ST", "AGG_CREDIT_TIER_2ND", "AGG_CREDIT_TIER_3RD", "AGG_CREDIT_TIER_4TH",
        "HIGH_INCOME_FLAG", "AGE_70_PLUS", "ROKU_FLAG", "BPS_PRESENT_FLAG", "NEW_MOVER_FLAG",
        "VETERAN_FLAG", "LIKELY_TO_MOVE_FLAG", "CARD_SPENDER_ANY", "HAS_ANY_LOAN",
        "SENIOR_AFFLUENT", "LIFE_EVENT_FLAG", "LIFE_INS_AFFINITY",
        "GENDER", "AGE_SEG", "INCOME_BAND", "BPS_TIER", "STABILITY_BUCKET",
        "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET",
    ]
    dummy = pd.DataFrame({c: [0] if c in [
        "AGE", "EST_HH_INCOME", "INCOME_LOG", "CREDIT_BEHAVIOR_SUM", "CREDIT_TIER_COUNT",
        "LENGTH_OF_RESIDENCE", "NEW_MOVER", "CREDIT_BEHAVIOR_ESTABLISHED", "CREDIT_BEHAVIOR_THRIVING",
        "CREDIT_BEHAVIOR_UP_AND_COMING", "CARD_SPENDERS_HIGH", "CARD_SPENDERS_LOW",
        "OPEN_PERSONAL_LOAN", "OPEN_STUDENT_LOAN", "VETERAN_IN_HOUSEHOLD", "LIKELY_TO_MOVE",
        "LIFE_INS_LOYALTY_HIGH_PROPENSITY", "LIFE_INS_LOYALTY_LOW_PROPENSITY",
        "AGG_CREDIT_TIER_1ST", "AGG_CREDIT_TIER_2ND", "AGG_CREDIT_TIER_3RD", "AGG_CREDIT_TIER_4TH",
        "HIGH_INCOME_FLAG", "AGE_70_PLUS", "ROKU_FLAG", "BPS_PRESENT_FLAG", "NEW_MOVER_FLAG",
        "VETERAN_FLAG", "LIKELY_TO_MOVE_FLAG", "CARD_SPENDER_ANY", "HAS_ANY_LOAN",
        "SENIOR_AFFLUENT", "LIFE_EVENT_FLAG", "LIFE_INS_AFFINITY",
    ] else [""] for c in base_cols})
    for c in ["GENDER", "AGE_SEG", "INCOME_BAND", "BPS_TIER", "STABILITY_BUCKET",
              "CREDIT_QUALITY_BUCKET", "LIFE_READINESS_BUCKET", "FINANCIAL_ENGAGEMENT_BUCKET"]:
        if c in dummy.columns:
            dummy[c] = "x"
    _, n, c = add_engineered_features(dummy)
    return (n, c)
