"""
Workflow Step 3: Exploratory Data Analysis (EDA)
Analyzes distributions, relationships, correlations, and saves visualizations.

Developed by Mahmood Mazloomi
For the Machine Learning course
Instructor: Dr. Nadi
University of Tehran – Kish Campus
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, TARGET_COL

HOTEL_ORDER = ["City Hotel", "Resort Hotel"]
DEPOSIT_ORDER = ["No Deposit", "Non Refund", "Refundable"]
HOTEL_PALETTE = {"City Hotel": "#2980b9", "Resort Hotel": "#e67e22"}
DATASET_PALETTE = {"Raw": "#34495e", "Deduplicated": "#16a085"}


def _summarize_deposit_by_hotel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate booking counts, cancellations, and cancellation rates by hotel and deposit type."""
    grouped = (
        df[df["hotel"].isin(HOTEL_ORDER)]
        .groupby(["hotel", "deposit_type"])[TARGET_COL]
        .agg(mean="mean", count="count", canceled="sum")
    )
    full_index = pd.MultiIndex.from_product([HOTEL_ORDER, DEPOSIT_ORDER], names=["hotel", "deposit_type"])
    return grouped.reindex(full_index, fill_value=0).reset_index()


def _annotate_bars(ax, formatter, offset: float, fontsize: int = 9):
    """Annotate every bar in all bar containers with a formatted label above the bar."""
    for container_idx, container in enumerate(ax.containers):
        for bar_idx, bar in enumerate(container):
            height = bar.get_height()
            if np.isnan(height):
                continue
            label = formatter(container_idx, bar_idx, height)
            if not label:
                continue
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2.0, height + offset),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=fontsize,
            )


def plot_deposit_cancellation_by_hotel(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Study deposit-type cancellation rates and booking volumes across City Hotel and Resort Hotel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _summarize_deposit_by_hotel(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    max_count = max(int(summary_df["count"].max()), 1)
    panel_specs = [
        {
            "ax": axes[0],
            "y_col": "mean",
            "title": "Cancellation Rate by Deposit Type & Hotel",
            "ylabel": "Cancellation Rate",
            "ylim": (0, 1.15),
            "offset": 0.025,
            "formatter": lambda _c, _b, v: f"{v * 100:.1f}%",
        },
        {
            "ax": axes[1],
            "y_col": "count",
            "title": "Booking Volume by Deposit Type & Hotel",
            "ylabel": "Booking Count",
            "ylim": (0, max_count * 1.15),
            "offset": max_count * 0.02,
            "formatter": lambda _c, _b, v: f"{int(round(v)):,}",
        },
    ]

    for spec in panel_specs:
        ax = spec["ax"]
        sns.barplot(
            data=summary_df,
            x="deposit_type",
            y=spec["y_col"],
            hue="hotel",
            order=DEPOSIT_ORDER,
            hue_order=HOTEL_ORDER,
            palette=HOTEL_PALETTE,
            ax=ax,
        )
        ax.set_title(spec["title"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Deposit Type", fontsize=11)
        ax.set_ylabel(spec["ylabel"], fontsize=11)
        ax.set_ylim(*spec["ylim"])
        ax.legend(title="Hotel Type")
        _annotate_bars(ax, spec["formatter"], spec["offset"])

    fig.suptitle("Deposit Type Cancellation Study: City Hotel vs Resort Hotel", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = output_dir / "eda_deposit_cancellation_by_hotel.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    study_summary = {hotel: {} for hotel in HOTEL_ORDER}
    for row in summary_df.itertuples(index=False):
        study_summary[row.hotel][row.deposit_type] = {
            "count": int(row.count),
            "canceled": int(row.canceled),
            "cancellation_rate": float(row.mean * 100),
        }
    return study_summary


def plot_deposit_raw_vs_dedup(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Compare raw vs deduplicated deposit cancellation rates and volumes across hotel types."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_summary = _summarize_deposit_by_hotel(df).assign(version="Raw")
    dedup_summary = _summarize_deposit_by_hotel(df.drop_duplicates()).assign(version="Deduplicated")
    combined = pd.concat([raw_summary, dedup_summary], ignore_index=True)
    combined["hotel_deposit"] = combined["hotel"].str.replace(" Hotel", "", regex=False) + "\n" + combined["deposit_type"]

    x_order = [
        f"{h.replace(' Hotel', '')}\n{d}"
        for h in HOTEL_ORDER
        for d in DEPOSIT_ORDER
    ]
    version_order = ["Raw", "Deduplicated"]
    max_count = max(int(combined["count"].max()), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    panel_specs = [
        {
            "ax": axes[0],
            "y_col": "mean",
            "title": "Cancellation Rate: Raw vs Deduplicated",
            "ylabel": "Cancellation Rate",
            "ylim": (0, 1.15),
            "offset": 0.025,
            "formatter": lambda _c, _b, v: f"{v * 100:.1f}%",
        },
        {
            "ax": axes[1],
            "y_col": "count",
            "title": "Booking Volume: Raw vs Deduplicated",
            "ylabel": "Booking Count",
            "ylim": (0, max_count * 1.18),
            "offset": max_count * 0.015,
            "formatter": lambda _c, _b, v: f"{int(round(v)):,}",
        },
    ]

    for spec in panel_specs:
        ax = spec["ax"]
        sns.barplot(
            data=combined,
            x="hotel_deposit",
            y=spec["y_col"],
            hue="version",
            order=x_order,
            hue_order=version_order,
            palette=DATASET_PALETTE,
            ax=ax,
        )
        ax.set_title(spec["title"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Hotel & Deposit Type", fontsize=11)
        ax.set_ylabel(spec["ylabel"], fontsize=11)
        ax.set_ylim(*spec["ylim"])
        ax.legend(title="Dataset")
        _annotate_bars(ax, spec["formatter"], spec["offset"], fontsize=8)

    fig.suptitle("Impact of Deduplication on Deposit Cancellation by Hotel", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = output_dir / "eda_deposit_raw_vs_dedup.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    comparison = {v: {h: {} for h in HOTEL_ORDER} for v in version_order}
    for row in combined.itertuples(index=False):
        comparison[row.version][row.hotel][row.deposit_type] = {
            "count": int(row.count),
            "canceled": int(row.canceled),
            "cancellation_rate": float(row.mean * 100),
        }
    return comparison


def _plot_non_refund_hotel_breakdown(nr_df: pd.DataFrame, group_col: str, ax, title: str, xlabel: str) -> dict:
    """Plot Non Refund booking volume and cancellation rate across a category split by hotel type."""
    order = nr_df[group_col].value_counts().index.tolist()
    stats = (
        nr_df.groupby([group_col, "hotel"])[TARGET_COL]
        .agg(count="count", mean="mean")
        .reindex(pd.MultiIndex.from_product([order, HOTEL_ORDER], names=[group_col, "hotel"]), fill_value=0)
        .reset_index()
    )
    rate_lookup = {
        (row.hotel, getattr(row, group_col)): row.mean * 100
        for row in stats.itertuples(index=False)
    }
    max_count = max(int(stats["count"].max()), 1)
    sns.barplot(
        data=stats,
        x=group_col,
        y="count",
        hue="hotel",
        order=order,
        hue_order=HOTEL_ORDER,
        palette=HOTEL_PALETTE,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Booking Count", fontsize=10)
    ax.set_ylim(0, max_count * 1.22)
    ax.legend(title="Hotel Type")
    _annotate_bars(
        ax,
        lambda c_idx, b_idx, h: (
            f"{int(round(h)):,}\n({rate_lookup[(HOTEL_ORDER[c_idx], order[b_idx])]:.1f}%)"
            if h > 0 else ""
        ),
        offset=max_count * 0.015,
        fontsize=8,
    )
    return {
        f"{row.hotel} | {getattr(row, group_col)}": {
            "count": int(row.count),
            "cancellation_rate": float(row.mean * 100),
        }
        for row in stats.itertuples(index=False)
        if row.count > 0
    }


def plot_non_refund_deep_dive(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Analyze why Non Refund bookings cancel at extreme rates (segments, countries, lead time, customer type)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nr_df = df[(df["deposit_type"] == "Non Refund") & (df["hotel"].isin(HOTEL_ORDER))].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Market Segment breakdown by Hotel
    seg_summary = _plot_non_refund_hotel_breakdown(
        nr_df,
        group_col="market_segment",
        ax=axes[0, 0],
        title="Non Refund Volume & Cancel Rate by Market Segment",
        xlabel="Market Segment",
    )

    # 2. Top 5 Origin Countries for Non Refund bookings
    top_countries = nr_df["country"].dropna().value_counts().head(5).index.tolist()
    country_stats = (
        nr_df[nr_df["country"].isin(top_countries)]
        .groupby("country")[TARGET_COL]
        .agg(count="count", mean="mean")
        .reindex(top_countries)
        .reset_index()
    )
    country_rates = {row.country: row.mean * 100 for row in country_stats.itertuples(index=False)}
    max_country = max(int(country_stats["count"].max()), 1)
    sns.barplot(
        data=country_stats,
        x="country",
        y="count",
        order=top_countries,
        color="#c0392b",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Top 5 Origin Countries for Non Refund Bookings", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Country Code", fontsize=10)
    axes[0, 1].set_ylabel("Booking Count", fontsize=10)
    axes[0, 1].set_ylim(0, max_country * 1.18)
    _annotate_bars(
        axes[0, 1],
        lambda _c, b_idx, h: f"{int(round(h)):,}\n({country_rates[top_countries[b_idx]]:.1f}%)",
        offset=max_country * 0.015,
        fontsize=8,
    )

    # 3. Lead Time Distribution by Deposit Type & Cancellation Status
    lead_df = df[df["deposit_type"].isin(DEPOSIT_ORDER)].copy()
    lead_df["status"] = lead_df[TARGET_COL].map({0: "Honored (0)", 1: "Canceled (1)"})
    status_order = ["Honored (0)", "Canceled (1)"]
    sns.boxplot(
        data=lead_df,
        x="deposit_type",
        y="lead_time",
        hue="status",
        order=DEPOSIT_ORDER,
        hue_order=status_order,
        palette={"Honored (0)": "#2ecc71", "Canceled (1)": "#e74c3c"},
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Lead Time by Deposit Type & Cancellation Status", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Deposit Type", fontsize=10)
    axes[1, 0].set_ylabel("Lead Time (Days, outliers hidden)", fontsize=10)
    axes[1, 0].legend(title="Booking Status")

    lead_medians = lead_df.groupby(["deposit_type", "status"])["lead_time"].median().to_dict()
    box_width = 0.4
    for d_idx, dep in enumerate(DEPOSIT_ORDER):
        for s_idx, status in enumerate(status_order):
            med = lead_medians.get((dep, status))
            if med is None or np.isnan(med):
                continue
            x_pos = d_idx + (-box_width / 2 if s_idx == 0 else box_width / 2)
            axes[1, 0].annotate(
                f"med: {med:.0f}d",
                (x_pos, med),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#1f2d3d",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

    # 4. Customer Type breakdown by Hotel for Non Refund bookings
    cust_summary = _plot_non_refund_hotel_breakdown(
        nr_df,
        group_col="customer_type",
        ax=axes[1, 1],
        title="Non Refund Volume & Cancel Rate by Customer Type",
        xlabel="Customer Type",
    )

    fig.suptitle("Non Refund Deposit Deep-Dive: Drivers of High Cancellation", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path = output_dir / "eda_non_refund_deep_dive.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "market_segment_by_hotel": seg_summary,
        "top_countries": {
            row.country: {"count": int(row.count), "cancellation_rate": float(row.mean * 100)}
            for row in country_stats.itertuples(index=False)
        },
        "lead_time_medians": {
            f"{dep} | {status}": float(val)
            for (dep, status), val in lead_medians.items()
            if not np.isnan(val)
        },
        "customer_type_by_hotel": cust_summary,
    }


def run_eda(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Perform exploratory data analysis and export key visual charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    insights = {}

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts()
    percentages = df[TARGET_COL].value_counts(normalize=True) * 100
    status_labels = counts.index.map({0: "Honored (0)", 1: "Canceled (1)"})
    sns.barplot(x=status_labels, y=counts.values, hue=status_labels, ax=ax, palette=["#2ecc71", "#e74c3c"], legend=False)
    ax.set_title("Booking Status Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(f"{height:,}\n({percentages.iloc[i]:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    target_path = output_dir / "eda_target_distribution.png"
    plt.savefig(target_path, dpi=300)
    plt.close()
    insights["cancellation_rate_overall"] = float(percentages.get(1, 0))

    # 2. Hotel Type vs Cancellation Rate
    fig, ax = plt.subplots(figsize=(7, 4))
    hotel_cancel = df.groupby("hotel")[TARGET_COL].mean().reset_index()
    sns.barplot(data=hotel_cancel, x="hotel", y=TARGET_COL, hue="hotel", ax=ax, palette="Blues_r", legend=False)
    ax.set_title("Cancellation Rate by Hotel Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cancellation Rate", fontsize=12)
    ax.set_ylim(0, 0.6)
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                    ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    hotel_path = output_dir / "eda_hotel_cancellation.png"
    plt.savefig(hotel_path, dpi=300)
    plt.close()

    # 3. Lead Time Distribution by Target
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df[df[TARGET_COL] == 0]["lead_time"].clip(upper=400),
                label="Honored (0)", fill=True, color="#2ecc71", alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df[TARGET_COL] == 1]["lead_time"].clip(upper=400),
                label="Canceled (1)", fill=True, color="#e74c3c", alpha=0.4, ax=ax)
    ax.set_title("Lead Time Distribution by Cancellation Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lead Time (Days, clipped at 400)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    plt.tight_layout()
    lead_time_path = output_dir / "eda_lead_time_distribution.png"
    plt.savefig(lead_time_path, dpi=300)
    plt.close()
    insights["lead_time_median_canceled"] = float(df[df[TARGET_COL] == 1]["lead_time"].median())
    insights["lead_time_median_honored"] = float(df[df[TARGET_COL] == 0]["lead_time"].median())

    # 4. Deposit Type vs Cancellation Rate
    fig, ax = plt.subplots(figsize=(7, 4))
    deposit_cancel = df.groupby("deposit_type")[TARGET_COL].agg(["mean", "count"]).reset_index()
    sns.barplot(data=deposit_cancel, x="deposit_type", y="mean", hue="deposit_type", ax=ax, palette="Reds_r", legend=False)
    ax.set_title("Cancellation Rate by Deposit Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cancellation Rate", fontsize=12)
    ax.set_ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.03),
                    ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    deposit_path = output_dir / "eda_deposit_cancellation.png"
    plt.savefig(deposit_path, dpi=300)
    plt.close()

    # 4b. Deposit Type vs Cancellation Study by Hotel Type (City Hotel vs Resort Hotel)
    insights["deposit_cancellation_by_hotel"] = plot_deposit_cancellation_by_hotel(df, output_dir=output_dir)

    # 4c. Raw vs Deduplicated Deposit Cancellation Comparison & Non Refund Deep-Dive
    insights["deposit_raw_vs_dedup"] = plot_deposit_raw_vs_dedup(df, output_dir=output_dir)
    insights["non_refund_study"] = plot_non_refund_deep_dive(df, output_dir=output_dir)

    # 5. Monthly Seasonality
    months_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    monthly = df.groupby("arrival_date_month")[TARGET_COL].agg(["mean", "count"]).reindex(months_order).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    sns.barplot(data=monthly, x="arrival_date_month", y="count", ax=ax1, color="#3498db", alpha=0.6)
    sns.lineplot(data=monthly, x="arrival_date_month", y="mean", ax=ax2, color="#e74c3c", marker="o", linewidth=2.5)
    ax1.tick_params(axis="x", labelrotation=45)
    plt.setp(ax1.get_xticklabels(), ha="right")
    ax1.set_title("Monthly Booking Volume vs Cancellation Rate", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Booking Count", fontsize=12, color="#2980b9")
    ax2.set_ylabel("Cancellation Rate", fontsize=12, color="#c0392b")
    ax2.set_ylim(0, 0.6)
    plt.tight_layout()
    monthly_path = output_dir / "eda_monthly_seasonality.png"
    plt.savefig(monthly_path, dpi=300)
    plt.close()

    # 6. Correlation Heatmap
    num_cols_for_corr = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
        "adr", "required_car_parking_spaces", "total_of_special_requests", TARGET_COL
    ]
    corr_matrix = df[num_cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", ax=ax, cbar=True, vmin=-0.3, vmax=0.6)
    ax.set_title("Correlation Heatmap of Numerical Features & Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    corr_path = output_dir / "eda_numerical_correlation.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()

    insights["top_correlations_with_target"] = corr_matrix[TARGET_COL].sort_values(ascending=False).to_dict()
    return insights

if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    insights = run_eda(df)
    print("EDA Completed successfully!")
    print("Insights:", insights)
