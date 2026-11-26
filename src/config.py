"""
Pipeline Configuration & Parameters
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "hotel_bookings.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Target Variable
TARGET_COL = "is_canceled"

# Data Leakage & Irrelevant Columns to Drop
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]