#!/usr/bin/env python3
"""
Run full pipeline: train on 22k.csv, then evaluate all models on validatemarch.csv
with the usual suite (confusion matrix, tier close rates, decile analysis, PNGs, CSVs).

Steps:
  1. Build training splits from 22k.csv -> training_tables/
  2. Train 5-tower model (train_multitower_sale_5towers.py)
  3. Prepare validatemarch: extract routing_transunion_raw from RAW_RESPONSE
  4. Run comprehensive_model_comparison.py --dataset validatemarch

Output:
  - exports/multitower_sale_5towers (updated from 22k training)
  - validatemarch_api_results.csv
  - comprehensive_model_comparison/validatemarch_* (scored CSVs, tier_close_rates, confusion_matrix, decile_analysis, PNGs)
  - comprehensive_model_comparison/summary_results.csv (appended with validatemarch rows)

Usage:
  python run_train_22k_validate_march.py
  python run_train_22k_validate_march.py --skip-training   # only prepare validatemarch and run evaluation (use existing models)
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIONAL = Path(__file__).resolve().parent


def run(cmd, description):
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, shell=False)
    if r.returncode != 0:
        print(f"Failed: {description}", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train on 22k, evaluate on validatemarch")
    parser.add_argument("--skip-training", action="store_true", help="Skip 22k splits and training; only prepare validatemarch and run evaluation")
    args = parser.parse_args()

    if not args.skip_training:
        run(
            [sys.executable, str(OPTIONAL / "build_training_splits_from_22k.py")],
            "Step 1: Build training splits from 22k.csv",
        )
        run(
            [sys.executable, str(REPO_ROOT / "train_multitower_sale_5towers.py")],
            "Step 2: Train 5-tower model on 22k",
        )
    else:
        print("Skipping training (--skip-training). Using existing exports.")

    run(
        [sys.executable, str(OPTIONAL / "prepare_validatemarch_for_eval.py")],
        "Step 3: Prepare validatemarch (extract routing_transunion_raw)",
    )
    run(
        [sys.executable, str(OPTIONAL / "comprehensive_model_comparison.py"), "--dataset", "validatemarch"],
        "Step 4: Run full evaluation suite on validatemarch",
    )

    print("\nDone. Check comprehensive_model_comparison/ for validatemarch_* PNGs and CSVs.")


if __name__ == "__main__":
    main()
