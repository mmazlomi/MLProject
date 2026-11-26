"""
Workflow Step 4: Data Quality Assessment

Evaluates:
- Duplicate records
- Missing values
- Invalid records (zero/negative guests, negative/extreme ADR)
- Outliers (IQR across continuous and count features)
- Potential target leakage

Cleaning is performed separately with a detailed audit report.

Developed by Mahmood Mazloomi
For the Machine Learning course
Instructor: Dr. Nadi
University of Tehran – Kish Campus
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import TARGET_COL, LEAKAGE_COLS

GUEST_COLS = ["adults", "children", "babies"]
EXTREME_ADR_THRESHOLD = 5000.0
CONTINUOUS_OUTLIER_COLS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive data quality report."""
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Cannot assess an empty DataFrame.")

    report: Dict[str, Any] = {}

    # 1. Dataset overview
    report["dataset"] = {
        "rows": n_rows,
        "columns": len(df.columns),
        "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)),
    }

    # 2. Duplicate records
    duplicate_count = int(df.duplicated().sum())
    report["duplicates"] = {
        "count": duplicate_count,
        "percentage": round(duplicate_count / n_rows * 100, 2),
    }

    # 3. Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    report["missing_values"] = {
        col: {
            "count": int(count),
            "percentage": round(count / n_rows * 100, 2),
        }
        for col, count in missing.items()
    }

    # 4. Data types
    report["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 5. Invalid records
    invalid: Dict[str, int] = {}
    if all(col in df.columns for col in GUEST_COLS):
        guest_values = df[GUEST_COLS].fillna(0)
        invalid["zero_guest_bookings"] = int(guest_values.eq(0).all(axis=1).sum())
        invalid["negative_guest_counts"] = int(guest_values.lt(0).any(axis=1).sum())

    if "adr" in df.columns:
        invalid["negative_adr"] = int(df["adr"].lt(0).sum())
        invalid["extreme_adr"] = int(df["adr"].ge(EXTREME_ADR_THRESHOLD).sum())

    if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
        total_nights = df["stays_in_weekend_nights"].fillna(0) + df["stays_in_week_nights"].fillna(0)
        invalid["zero_night_bookings"] = int(total_nights.le(0).sum())

    report["invalid_records"] = invalid

    # 6. Outlier detection (IQR on continuous/count features only)
    outliers: Dict[str, Dict[str, float | int]] = {}
    for col in CONTINUOUS_OUTLIER_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())

        outliers[col] = {
            "count": count,
            "percentage": round(count / len(series) * 100, 2),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }

    report["outliers"] = outliers

    # 7. Potential leakage columns
    report["potential_leakage_columns"] = [col for col in LEAKAGE_COLS if col in df.columns]

    # 8. Target distribution
    if TARGET_COL in df.columns:
        report["target_distribution"] = df[TARGET_COL].value_counts(normalize=True).to_dict()

    return report


def clean_data_quality_issues(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean confirmed invalid records, leakage columns, and optional duplicates.

    Missing values and non-anomalous IQR outliers are preserved for downstream preprocessing.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    audit: Dict[str, Any] = {}

    # 1. Remove potential leakage columns strictly
    leakage_cols = [col for col in LEAKAGE_COLS if col in df_clean.columns]
    if leakage_cols:
        df_clean = df_clean.drop(columns=leakage_cols)
    audit["leakage_columns_removed"] = leakage_cols

    # 2. Build combined invalid-record mask (avoid intermediate copies)
    invalid_mask = pd.Series(False, index=df_clean.index)

    if all(col in df_clean.columns for col in GUEST_COLS):
        guest_values = df_clean[GUEST_COLS].fillna(0)
        zero_guest_mask = guest_values.eq(0).all(axis=1)
        negative_guest_mask = guest_values.lt(0).any(axis=1)
        audit["zero_guest_removed"] = int(zero_guest_mask.sum())
        audit["negative_guest_removed"] = int(negative_guest_mask.sum())
        invalid_mask |= zero_guest_mask | negative_guest_mask

    if "adr" in df_clean.columns:
        negative_adr_mask = df_clean["adr"].lt(0)
        extreme_adr_mask = df_clean["adr"].ge(EXTREME_ADR_THRESHOLD)
        audit["negative_adr_removed"] = int(negative_adr_mask.sum())
        audit["extreme_adr_removed"] = int(extreme_adr_mask.sum())
        invalid_mask |= negative_adr_mask | extreme_adr_mask

    if "stays_in_weekend_nights" in df_clean.columns and "stays_in_week_nights" in df_clean.columns:
        total_nights = df_clean["stays_in_weekend_nights"].fillna(0) + df_clean["stays_in_week_nights"].fillna(0)
        zero_night_mask = total_nights.le(0)
        audit["zero_night_removed"] = int(zero_night_mask.sum())
        invalid_mask |= zero_night_mask

    df_clean = df_clean.loc[~invalid_mask].copy()

    # 3. Optional duplicate removal on feature columns only (avoids biasing conflicting-label pairs)
    feature_cols = [c for c in df_clean.columns if c != TARGET_COL]
    duplicate_count = int(df_clean.duplicated(subset=feature_cols).sum())
    audit["post_cleaning_duplicate_candidates"] = duplicate_count
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates(subset=feature_cols, keep="first").copy()
    audit["duplicates_removed"] = duplicate_count if drop_duplicates else 0

    # 4. Final audit summary
    audit["initial_rows"] = initial_rows
    audit["final_rows"] = len(df_clean)
    audit["total_rows_removed"] = initial_rows - len(df_clean)
    audit["drop_duplicates"] = drop_duplicates

    return df_clean, audit


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw = load_raw_data()
    report = assess_data_quality(raw)
    print("=== DATA QUALITY ASSESSMENT ===")
    for key, value in report.items():
        print(f"\n{key}:")
        print(value)

    clean_df, audit = clean_data_quality_issues(raw, drop_duplicates=True)
    print("\n=== CLEANING AUDIT ===")
    for key, value in audit.items():
        print(f"{key}: {value}")
