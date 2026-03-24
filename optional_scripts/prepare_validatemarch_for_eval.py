#!/usr/bin/env python3
"""
Prepare validatemarch.csv for model evaluation: extract routing_transunion_raw from RAW_RESPONSE
so comprehensive_model_comparison can rescore with each model.

Usage:
  python prepare_validatemarch_for_eval.py
  python prepare_validatemarch_for_eval.py --input validatemarch.csv --output validatemarch_api_results.csv

Writes:
  validatemarch_api_results.csv (original columns + routing_transunion_raw, CLEAN_PHONE_NUMBER, SALE_MADE_BINARY)
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_transunion_raw(raw_response):
    """Parse RAW_RESPONSE JSON and return routing.transunion_raw as JSON string, or None."""
    if pd.isna(raw_response) or (isinstance(raw_response, str) and raw_response.strip() == ""):
        return None
    if "no transunion data" in str(raw_response).lower():
        return None
    try:
        data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    except json.JSONDecodeError:
        return None
    routing = data.get("routing") or {}
    tu_raw = routing.get("transunion_raw")
    if tu_raw is None:
        return None
    return json.dumps(tu_raw) if isinstance(tu_raw, dict) else tu_raw


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare validatemarch for evaluation")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "validatemarch.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "validatemarch_api_results.csv")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    n = len(df)

    if "RAW_RESPONSE" not in df.columns:
        print("Error: RAW_RESPONSE column not found.", file=sys.stderr)
        sys.exit(1)

    routing_tu_raw = df["RAW_RESPONSE"].map(extract_transunion_raw)
    df = df.copy()
    df["routing_transunion_raw"] = routing_tu_raw

    if "CLEAN_PHONE_NUMBER" not in df.columns and "INPUT_PHONE" in df.columns:
        df["CLEAN_PHONE_NUMBER"] = df["INPUT_PHONE"].astype(str)

    if "SALE_MADE_BINARY" not in df.columns and "SALE_MADE_FLAG" in df.columns:
        raw = df["SALE_MADE_FLAG"].astype(str).str.strip().str.upper()
        df["SALE_MADE_BINARY"] = (raw.isin(("Y", "1", "TRUE", "YES")) | (df["SALE_MADE_FLAG"] == 1)).astype(int)

    n_with_tu = df["routing_transunion_raw"].notna() & (df["routing_transunion_raw"].astype(str).str.strip() != "")
    n_with_tu = n_with_tu.sum()
    print(f"  Rows with TransUnion data: {n_with_tu:,} / {n:,}")

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print("Next: python comprehensive_model_comparison.py --dataset validatemarch")


if __name__ == "__main__":
    main()
