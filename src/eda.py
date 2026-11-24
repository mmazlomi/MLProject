
"""
Workflow Step 3: Exploratory Data Analysis (EDA)
Analyzes distributions, relationships, correlations, and saves visualizations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL


def run_eda(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Perform exploratory data analysis and export key visual charts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    insights = {}

    # ---------------------------------------------------------
    # 1. Target Distribution
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 4))

    counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    percentages = df[TARGET_COL].value_counts(normalize=True) * 100

    target_labels = ["Honored (0)", "Canceled (1)"]

    sns.barplot(
        x=target_labels,
        y=counts.values,
        hue=target_labels,
        hue_order=target_labels,
        palette=["#2ecc71", "#e74c3c"],
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Booking Status Distribution",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Count", fontsize=12)

    for i, p in enumerate(ax.patches):
        height = p.get_height()

        percentage = percentages.get(i, 0)

        ax.annotate(
            f"{height:,.0f}\n({percentage:.1f}%)",
            (p.get_x() + p.get_width() / 2., height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11
        )

    plt.tight_layout()

    target_path = output_dir / "eda_target_distribution.png"
    plt.savefig(target_path, dpi=300)
    plt.close(fig)

    insights["cancellation_rate_overall"] = float(percentages.get(1, 0))

    # ---------------------------------------------------------
    # 2. Hotel Type vs Cancellation Rate
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    hotel_cancel = (
        df.groupby("hotel", dropna=False)[TARGET_COL]
        .mean()
        .reset_index()
    )

    sns.barplot(
        data=hotel_cancel,
        x="hotel",
        y=TARGET_COL,
        hue="hotel",
        palette="Blues_r",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Cancellation Rate by Hotel Type",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Cancellation Rate", fontsize=12)
    ax.set_ylim(0, 0.6)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() * 100:.1f}%",
            (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()

    hotel_path = output_dir / "eda_hotel_cancellation.png"
    plt.savefig(hotel_path, dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # 3. Lead Time Distribution by Target
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.kdeplot(
        data=df[df[TARGET_COL] == 0]["lead_time"].clip(upper=400),
        label="Honored (0)",
        fill=True,
        color="#2ecc71",
        alpha=0.4,
        ax=ax
    )

    sns.kdeplot(
        data=df[df[TARGET_COL] == 1]["lead_time"].clip(upper=400),
        label="Canceled (1)",
        fill=True,
        color="#e74c3c",
        alpha=0.4,
        ax=ax
    )

    ax.set_title(
        "Lead Time Distribution by Cancellation Status",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Lead Time (Days, clipped at 400)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.legend()

    plt.tight_layout()

    lead_time_path = output_dir / "eda_lead_time_distribution.png"
    plt.savefig(lead_time_path, dpi=300)
    plt.close(fig)

    insights["lead_time_median_canceled"] = float(
        df[df[TARGET_COL] == 1]["lead_time"].median()
    )

    insights["lead_time_median_honored"] = float(
        df[df[TARGET_COL] == 0]["lead_time"].median()
    )

    # ---------------------------------------------------------
    # 4. Deposit Type vs Cancellation Rate
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    deposit_cancel = (
        df.groupby("deposit_type", dropna=False)[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
    )

    sns.barplot(
        data=deposit_cancel,
        x="deposit_type",
        y="mean",
        hue="deposit_type",
        palette="Reds_r",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Cancellation Rate by Deposit Type",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Cancellation Rate", fontsize=12)
    ax.set_ylim(0, 1.1)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() * 100:.1f}%",
            (p.get_x() + p.get_width() / 2., p.get_height() + 0.03),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()

    deposit_path = output_dir / "eda_deposit_cancellation.png"
    plt.savefig(deposit_path, dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # 5. Monthly Seasonality
    # ---------------------------------------------------------

    months_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly = (
        df.groupby("arrival_date_month", observed=False)[TARGET_COL]
        .agg(["mean", "count"])
        .reindex(months_order)
        .rename_axis("arrival_date_month")
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax2 = ax1.twinx()

    sns.barplot(
        data=monthly,
        x="arrival_date_month",
        y="count",
        order=months_order,
        color="#3498db",
        alpha=0.6,
        ax=ax1
    )

    sns.lineplot(
        data=monthly,
        x="arrival_date_month",
        y="mean",
        sort=False,
        color="#e74c3c",
        marker="o",
        linewidth=2.5,
        ax=ax2
    )

    ax1.tick_params(axis="x", labelrotation=45)

    ax1.set_title(
        "Monthly Booking Volume vs Cancellation Rate",
        fontsize=14,
        fontweight="bold"
    )

    ax1.set_ylabel("Booking Count", fontsize=12, color="#2980b9")
    ax2.set_ylabel("Cancellation Rate", fontsize=12, color="#c0392b")

    ax2.set_ylim(0, 0.6)

    plt.tight_layout()

    monthly_path = output_dir / "eda_monthly_seasonality.png"
    plt.savefig(monthly_path, dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # 6. Correlation Heatmap
    # ---------------------------------------------------------

    num_cols_for_corr = [
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
        TARGET_COL
    ]

    corr_matrix = df[num_cols_for_corr].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar=True,
        vmin=-0.3,
        vmax=0.6
    )

    ax.set_title(
        "Correlation Heatmap of Numerical Features & Target",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    corr_path = output_dir / "eda_numerical_correlation.png"
    plt.savefig(corr_path, dpi=300)
    plt.close(fig)

    insights["top_correlations_with_target"] = (
        corr_matrix[TARGET_COL]
        .sort_values(ascending=False)
        .to_dict()
    )

    return insights


if __name__ == "__main__":

    from src.data_loader import load_raw_data

    df = load_raw_data()

    insights = run_eda(df)

    print("EDA Completed successfully!")
    print("Insights:", insights)
