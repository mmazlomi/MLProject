"""
Master Machine Learning Pipeline Runner
Executes all 13 Workflow Stages End-to-End:
1. Problem Definition
2. Data Collection
3. Exploratory Data Analysis (EDA)

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import numpy as np
import pandas as pd
from src.config import (
    DATA_PATH,
    ARTIFACTS_DIR,
    FIGURES_DIR
)
from src.data_loader import load_raw_data, get_data_summary
from src.eda import run_eda


def print_stage_header(stage_num: int, title: str):
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"STAGE {stage_num}: {title.upper()}")
    print(f"{separator}\n")

def run_ml_pipeline():
    total_start = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Problem Definition
    # -------------------------------------------------------------
    print_stage_header(1, "Problem Definition")
    problem_def = {
        "Objective": "Predict the likelihood of hotel reservation cancellations prior to arrival.",
        "Target": "is_canceled (0 = Booking Honored, 1 = Booking Canceled)",
        "Business Goal": "Empower hotel revenue management to mitigate cancellation revenue loss, optimize overbooking thresholds, and protect high-value customer relationships.",
        "Evaluation Criteria": "ROC-AUC (primary ranking metric), F1-Score (balance of Precision & Recall), Recall on Class 1 (catching cancellations early), and Precision (avoiding false alarms)."
    }
    for k, v in problem_def.items():
        print(f"  [{k}]: {v}")

    # -------------------------------------------------------------
    # 2. Data Collection
    # -------------------------------------------------------------
    print_stage_header(2, "Data Collection")
    raw_df = load_raw_data(DATA_PATH)
    data_summary = get_data_summary(raw_df)
    print(f"Loaded raw dataset from: {DATA_PATH}")
    print(f"Dataset Dimensions: {data_summary['n_rows']:,} rows, {data_summary['n_cols']} columns")
    print(f"Memory Footprint: {data_summary['memory_usage_mb']:.2f} MB")
    print(f"Raw Target Distribution: {data_summary['target_distribution']}")

    # -------------------------------------------------------------
    # 3. Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    print_stage_header(3, "Exploratory Data Analysis (EDA)")
    eda_insights = run_eda(raw_df, output_dir=FIGURES_DIR)
    print("Generated and saved EDA visual figures:")
    print("  - artifacts/figures/eda_target_distribution.png")
    print("  - artifacts/figures/eda_hotel_cancellation.png")
    print("  - artifacts/figures/eda_lead_time_distribution.png")
    print("  - artifacts/figures/eda_deposit_cancellation.png")
    print("  - artifacts/figures/eda_monthly_seasonality.png")
    print("  - artifacts/figures/eda_numerical_correlation.png")
    print(f"Key Insight: Overall raw cancellation rate = {eda_insights['cancellation_rate_overall']:.1f}%")
    print(f"Key Insight: Canceled bookings have median lead time of {eda_insights['lead_time_median_canceled']:.0f} days vs {eda_insights['lead_time_median_honored']:.0f} days for honored bookings.")

  

if __name__ == "__main__":
    run_ml_pipeline()
