"""
Workflow Step 2: Data Collection
Loads raw data and reports basic structural properties.

Developed by Mahmood Mazloomi
For the Machine Learning course
Instructor: Dr. Nadi
University of Tehran – Kish Campus
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from typing import Tuple, Dict, Any
from src.config import DATA_PATH, TARGET_COL

def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw hotel bookings CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    df = pd.read_csv(path)
    return df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract structural summary metrics from dataframe."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "target_distribution": df[TARGET_COL].value_counts(normalize=True).to_dict() if TARGET_COL in df.columns else {},
        "missing_summary": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    }
    return summary

if __name__ == "__main__":
    df = load_raw_data()
    summary = get_data_summary(df)
    print("=== Data Summary ===")
    print(f"Rows: {summary['n_rows']}, Cols: {summary['n_cols']}")
    print(f"Memory: {summary['memory_usage_mb']:.2f} MB")
    print("Missing columns:", summary['missing_summary'])
    print("Target distribution:", summary['target_distribution'])
