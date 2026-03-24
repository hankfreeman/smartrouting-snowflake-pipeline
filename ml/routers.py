"""
Router definitions for optimization.
A feature is used either for ROUTING (bucketing into segments) or as a TRAINING feature (model input), never both.
Each router has: router_id, routing_dimensions (list of column names), segments (list of (name, filter_dict)).
filter_dict = {dim: value} so we only train on rows that match; training features = FEATURE_COLS - routing_dimensions.
"""
import pandas as pd

from .config import FEATURE_COLS

# (router_id, routing_dimensions, segments)
# routing_dimensions: list of column names used to bucket; these are excluded from model input
# segments: list of (segment_name, filter_dict) where filter_dict maps each routing dim to value

ROUTER_12 = "router_12"
ROUTING_12 = ["GENDER", "LEAD_SOURCE_SEG", "AGE_SEG"]
SEGMENTS_12 = [
    ("M_R_A1", {"GENDER": "M", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A1"}),
    ("M_R_A2", {"GENDER": "M", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A2"}),
    ("M_R_A3", {"GENDER": "M", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A3"}),
    ("M_N_A1", {"GENDER": "M", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A1"}),
    ("M_N_A2", {"GENDER": "M", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A2"}),
    ("M_N_A3", {"GENDER": "M", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A3"}),
    ("F_R_A1", {"GENDER": "F", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A1"}),
    ("F_R_A2", {"GENDER": "F", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A2"}),
    ("F_R_A3", {"GENDER": "F", "LEAD_SOURCE_SEG": "R", "AGE_SEG": "A3"}),
    ("F_N_A1", {"GENDER": "F", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A1"}),
    ("F_N_A2", {"GENDER": "F", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A2"}),
    ("F_N_A3", {"GENDER": "F", "LEAD_SOURCE_SEG": "N", "AGE_SEG": "A3"}),
]

ROUTER_1 = "router_1"
ROUTING_1 = []
SEGMENTS_1 = [("GLOBAL", {})]

# Simple single model: no routing, features = gender, Roku flag, log(age), log income, BPS ability (A/B/C/D).
ROUTER_SIMPLE_GLOBAL = "router_simple_global"
ROUTING_SIMPLE = []
SEGMENTS_SIMPLE = [("GLOBAL", {})]
FIXED_FEATURES_SIMPLE = ["GENDER", "ROKU_FLAG", "AGE_LOG", "INCOME_LOG", "BPS_ABILITY_LETTER"]

ROUTER_6_GENDER_AGE = "router_6_gender_age"
ROUTING_6_GENDER_AGE = ["GENDER", "AGE_SEG"]
SEGMENTS_6_GENDER_AGE = [
    ("M_A1", {"GENDER": "M", "AGE_SEG": "A1"}),
    ("M_A2", {"GENDER": "M", "AGE_SEG": "A2"}),
    ("M_A3", {"GENDER": "M", "AGE_SEG": "A3"}),
    ("F_A1", {"GENDER": "F", "AGE_SEG": "A1"}),
    ("F_A2", {"GENDER": "F", "AGE_SEG": "A2"}),
    ("F_A3", {"GENDER": "F", "AGE_SEG": "A3"}),
]

ROUTER_6_LEAD_AGE = "router_6_lead_age"
ROUTING_6_LEAD_AGE = ["LEAD_SOURCE_SEG", "AGE_SEG"]
SEGMENTS_6_LEAD_AGE = [
    ("R_A1", {"LEAD_SOURCE_SEG": "R", "AGE_SEG": "A1"}),
    ("R_A2", {"LEAD_SOURCE_SEG": "R", "AGE_SEG": "A2"}),
    ("R_A3", {"LEAD_SOURCE_SEG": "R", "AGE_SEG": "A3"}),
    ("N_A1", {"LEAD_SOURCE_SEG": "N", "AGE_SEG": "A1"}),
    ("N_A2", {"LEAD_SOURCE_SEG": "N", "AGE_SEG": "A2"}),
    ("N_A3", {"LEAD_SOURCE_SEG": "N", "AGE_SEG": "A3"}),
]

ROUTER_3_AGE = "router_3_age"
ROUTING_3_AGE = ["AGE_SEG"]
SEGMENTS_3_AGE = [
    ("A1", {"AGE_SEG": "A1"}),
    ("A2", {"AGE_SEG": "A2"}),
    ("A3", {"AGE_SEG": "A3"}),
]

ROUTER_4_GENDER_LEAD = "router_4_gender_lead"
ROUTING_4_GENDER_LEAD = ["GENDER", "LEAD_SOURCE_SEG"]
SEGMENTS_4_GENDER_LEAD = [
    ("M_R", {"GENDER": "M", "LEAD_SOURCE_SEG": "R"}),
    ("M_N", {"GENDER": "M", "LEAD_SOURCE_SEG": "N"}),
    ("F_R", {"GENDER": "F", "LEAD_SOURCE_SEG": "R"}),
    ("F_N", {"GENDER": "F", "LEAD_SOURCE_SEG": "N"}),
]

# Route by lead source + gender; each segment model uses only AGE and EST_HH_INCOME (no RFE).
ROUTER_4_LEAD_GENDER_AGE_INCOME = "router_4_lead_gender_age_income"
ROUTING_4_LEAD_GENDER = ["LEAD_SOURCE_SEG", "GENDER"]
SEGMENTS_4_LEAD_GENDER = [
    ("M_R", {"GENDER": "M", "LEAD_SOURCE_SEG": "R"}),
    ("M_N", {"GENDER": "M", "LEAD_SOURCE_SEG": "N"}),
    ("F_R", {"GENDER": "F", "LEAD_SOURCE_SEG": "R"}),
    ("F_N", {"GENDER": "F", "LEAD_SOURCE_SEG": "N"}),
]
FIXED_FEATURES_4_LEAD_GENDER_AGE_INCOME = ["AGE", "EST_HH_INCOME"]

# Lead source first; then if BPS present -> BPS wealth (low/medium/high) × ability (great/good/fair/bad);
# if no BPS -> route by INCOME_BAND. Features for all segments: GENDER, AGE.
ROUTER_LEAD_BPS_OR_INCOME = "router_lead_bps_or_income"
_ROUTING_LEAD_BPS = ["LEAD_SOURCE_SEG", "BPS_WEALTH", "BPS_ABILITY", "INCOME_BAND"]
_BPS_WEALTH = ["low", "medium", "high"]
_BPS_ABILITY = ["great", "good", "fair", "bad"]
_INCOME_BANDS_6 = ["0_25", "25_50", "50_75", "75_100", "100_150", "150_plus"]
SEGMENTS_LEAD_BPS = [
    (f"{lead}_BPS_{w}_{a}", {"LEAD_SOURCE_SEG": lead, "BPS_WEALTH": w, "BPS_ABILITY": a})
    for lead in ("R", "N")
    for w in _BPS_WEALTH
    for a in _BPS_ABILITY
]
SEGMENTS_LEAD_INCOME = [
    (f"{lead}_INCOME_{ib}", {"LEAD_SOURCE_SEG": lead, "BPS_WEALTH": "NO_BPS", "BPS_ABILITY": "NO_BPS", "INCOME_BAND": ib})
    for lead in ("R", "N")
    for ib in _INCOME_BANDS_6
]
SEGMENTS_LEAD_BPS_OR_INCOME = SEGMENTS_LEAD_BPS + SEGMENTS_LEAD_INCOME
FIXED_FEATURES_LEAD_BPS_OR_INCOME = ["GENDER", "AGE"]

# 72 segments: gender × lead_source × age_band × income_band (6 bands)
ROUTER_72_GENDER_AGE_LEAD_INCOME = "router_72_gender_age_lead_income"
ROUTING_72 = ["GENDER", "LEAD_SOURCE_SEG", "AGE_SEG", "INCOME_BAND"]
_INCOME_BANDS = ["0_25", "25_50", "50_75", "75_100", "100_150", "150_plus"]
SEGMENTS_72 = [
    (f"{g}_{l}_{a}_{ib}", {"GENDER": g, "LEAD_SOURCE_SEG": l, "AGE_SEG": a, "INCOME_BAND": ib})
    for g in ("M", "F")
    for l in ("R", "N")
    for a in ("A1", "A2", "A3")
    for ib in _INCOME_BANDS
]

# 24 segments: lead_source × gender × 6 age bands (55-60 .. 80-85). Use RFE (all non-routing features) for better models.
ROUTER_24_LEAD_GENDER_AGE_6BAND = "router_24_lead_gender_age_6band"
ROUTING_24 = ["GENDER", "LEAD_SOURCE_SEG", "AGE_SEG"]
_AGE_6 = ["A1", "A2", "A3", "A4", "A5", "A6"]  # 55-60, 60-65, 65-70, 70-75, 75-80, 80-85
SEGMENTS_24 = [
    (f"{g}_{l}_{a}", {"GENDER": g, "LEAD_SOURCE_SEG": l, "AGE_SEG": a})
    for g in ("M", "F")
    for l in ("R", "N")
    for a in _AGE_6
]
FIXED_FEATURES_24 = None  # use RFE over all non-routing features (was single EST_HH_INCOME)

# Simple BPS vs non-BPS: two segments. BPS segment uses gender, roku, age, bps; NO_BPS uses gender, roku, age, est_hh_income.
ROUTER_BPS_VS_NO_BPS = "router_bps_vs_no_bps"
ROUTING_BPS_PRESENT = ["BPS_PRESENT"]
SEGMENTS_BPS_VS_NO_BPS = [
    ("BPS", {"BPS_PRESENT": "Y"}),
    ("NO_BPS", {"BPS_PRESENT": "N"}),
]
FIXED_FEATURES_BPS = ["GENDER", "ROKU_FLAG", "AGE", "BPS"]
FIXED_FEATURES_NO_BPS = ["GENDER", "ROKU_FLAG", "AGE", "EST_HH_INCOME"]
FIXED_FEATURES_PER_SEGMENT_BPS_VS_NO_BPS = {
    "BPS": FIXED_FEATURES_BPS,
    "NO_BPS": FIXED_FEATURES_NO_BPS,
}


def get_all_routers():
    """Return list of (router_id, routing_dimensions, segments, fixed_training_features or None, fixed_features_per_segment or None)."""
    return [
        (ROUTER_12, ROUTING_12, SEGMENTS_12, None, None),
        (ROUTER_1, ROUTING_1, SEGMENTS_1, None, None),
        (ROUTER_SIMPLE_GLOBAL, ROUTING_SIMPLE, SEGMENTS_SIMPLE, FIXED_FEATURES_SIMPLE, None),
        (ROUTER_6_GENDER_AGE, ROUTING_6_GENDER_AGE, SEGMENTS_6_GENDER_AGE, None, None),
        (ROUTER_6_LEAD_AGE, ROUTING_6_LEAD_AGE, SEGMENTS_6_LEAD_AGE, None, None),
        (ROUTER_3_AGE, ROUTING_3_AGE, SEGMENTS_3_AGE, None, None),
        (ROUTER_4_GENDER_LEAD, ROUTING_4_GENDER_LEAD, SEGMENTS_4_GENDER_LEAD, None, None),
        (ROUTER_4_LEAD_GENDER_AGE_INCOME, ROUTING_4_LEAD_GENDER, SEGMENTS_4_LEAD_GENDER, FIXED_FEATURES_4_LEAD_GENDER_AGE_INCOME, None),
        (ROUTER_LEAD_BPS_OR_INCOME, _ROUTING_LEAD_BPS, SEGMENTS_LEAD_BPS_OR_INCOME, FIXED_FEATURES_LEAD_BPS_OR_INCOME, None),
        (ROUTER_72_GENDER_AGE_LEAD_INCOME, ROUTING_72, SEGMENTS_72, None, None),
        (ROUTER_24_LEAD_GENDER_AGE_6BAND, ROUTING_24, SEGMENTS_24, None, None),
        (ROUTER_BPS_VS_NO_BPS, ROUTING_BPS_PRESENT, SEGMENTS_BPS_VS_NO_BPS, None, FIXED_FEATURES_PER_SEGMENT_BPS_VS_NO_BPS),
    ]


def get_router_spec(router_id):
    """Return (routing_dimensions, segments, fixed_training_features or None, fixed_features_per_segment or None) for router_id."""
    for row in get_all_routers():
        rid, rdim, segs, fixed, fixed_per_seg = row[0], row[1], row[2], row[3], row[4]
        if rid == router_id:
            return rdim, segs, fixed, fixed_per_seg
    return None, None, None, None


def get_training_feature_cols(router_id, segment_name=None):
    """Features used as model input. If segment_name given and router has fixed_features_per_segment, use that; else fixed_training_features; else FEATURE_COLS - routing_dims."""
    routing_dims, _, fixed, fixed_per_segment = get_router_spec(router_id)
    if segment_name and fixed_per_segment and segment_name in fixed_per_segment:
        return list(fixed_per_segment[segment_name])
    if fixed is not None:
        return list(fixed)
    if routing_dims is None:
        return list(FEATURE_COLS)
    return [c for c in FEATURE_COLS if c not in routing_dims]


def get_training_feature_cols_for_dims(routing_dims):
    """Training feature columns when using this list of routing dimensions (excluded from model input)."""
    if not routing_dims:
        return list(FEATURE_COLS)
    return [c for c in FEATURE_COLS if c not in routing_dims]


def get_segment_name_for_row_given_spec(routing_dims, segments, row):
    """Return segment name for this row given (routing_dims, segments) without router_id."""
    if not segments:
        return None
    if not routing_dims:
        return segments[0][0]
    for name, filter_dict in segments:
        match = True
        for dim in routing_dims:
            row_val = str(row.get(dim, "")).strip().upper() or ""
            seg_val = str(filter_dict.get(dim, "")).strip().upper()
            if seg_val and row_val != seg_val:
                match = False
                break
        if match:
            return name
    return None


def get_segment_mask(df, segment_spec, routing_dimensions):
    """
    segment_spec = (name, filter_dict). filter_dict keys = routing_dimensions.
    Returns boolean series of rows that belong to this segment.
    """
    name, filter_dict = segment_spec
    if not routing_dimensions:
        return pd.Series(True, index=df.index)
    mask = True
    for dim in routing_dimensions:
        val = filter_dict.get(dim)
        if val is None:
            continue
        col = df[dim].astype(str).str.strip().str.upper()
        mask = mask & (col == str(val).strip().upper())
    return mask


def get_segment_name_for_row(router_id, row):
    """Return the segment name for this row under the given router."""
    routing_dims, segments, _, _ = get_router_spec(router_id)
    if segments is None:
        return None
    if not routing_dims:
        return segments[0][0]
    for name, filter_dict in segments:
        match = True
        for dim in routing_dims:
            row_val = str(row.get(dim, "")).strip().upper() or ""
            seg_val = str(filter_dict.get(dim, "")).strip().upper()
            if seg_val and row_val != seg_val:
                match = False
                break
        if match:
            return name
    return None
