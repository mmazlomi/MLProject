import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PRICE (ADR) & STAY DURATION ANOMALIES
# ==========================================
def plot_price_and_stay_anomalies(df: pd.DataFrame, anomaly_dir: str):
    """
    Visualizes negative/extreme ADR outliers and zero-night / zero-price anomalies.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        '1. Financial & Stay Duration Anomalies (ADR & Zero-Night Bugs)', 
        fontsize=14, fontweight='bold'
    )

    # Subplot 1: ADR Scatter vs Total Nights (Highlighting -6.38 and 5400 outliers)
    total_nights = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    normal_mask = (df['adr'] >= 0) & (df['adr'] <= 1000)
    
    axes[0].scatter(
        total_nights[normal_mask], df.loc[normal_mask, 'adr'], 
        alpha=0.3, color='steelblue', label='Normal (0 <= ADR <= 1000)', s=15
    )
    axes[0].scatter(
        total_nights[~normal_mask], df.loc[~normal_mask, 'adr'], 
        color='crimson', s=90, zorder=5, label='Extreme Outliers (-6.38 & 5400)'
    )
    for idx in df[~normal_mask].index:
        axes[0].annotate(
            f"ADR: {df.loc[idx, 'adr']}", 
            (total_nights[idx], df.loc[idx, 'adr']),
            xytext=(10, -10), textcoords='offset points',
            fontweight='bold', color='crimson'
        )
    axes[0].set_title("ADR vs. Total Nights (Exposing 5400 & -6.38)")
    axes[0].set_xlabel("Total Stay Nights")
    axes[0].set_ylabel("ADR (EUR)")
    axes[0].legend()

    # Subplot 2: Breakdown of Zero/Negative ADR Bookings (ADR <= 0)
    zero_adr_df = df[df['adr'] <= 0].copy()
    zero_adr_df['Status'] = np.where(zero_adr_df['is_canceled'] == 1, 'Canceled', 'Checked-Out (Active)')
    vc_zero = zero_adr_df['Status'].value_counts()
    sns.barplot(x=vc_zero.index, y=vc_zero.values, ax=axes[1], palette=['seagreen', 'coral'], hue=vc_zero.index, legend=False)
    axes[1].set_title(f"Bookings with ADR <= 0 (Total: {len(zero_adr_df)} rows)")
    axes[1].set_ylabel("Row Count")
    for i, v in enumerate(vc_zero.values):
        axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

    # Subplot 3: Zero Total Nights vs Cancellation Status
    zero_nights_df = df[total_nights == 0].copy()
    zero_nights_df['Category'] = np.where(
        zero_nights_df['is_canceled'] == 1, 
        'Canceled (0 Nights)', 
        'Checked-Out with 0 Nights\n(Day-Use / System Bug)'
    )
    vc_nights = zero_nights_df['Category'].value_counts()
    sns.barplot(x=vc_nights.index, y=vc_nights.values, ax=axes[2], palette=['darkorange', 'crimson'], hue=vc_nights.index, legend=False)
    axes[2].set_title(f"Bookings with 0 Total Nights (Total: {len(zero_nights_df)} rows)")
    axes[2].set_ylabel("Row Count")
    for i, v in enumerate(vc_nights.values):
        axes[2].text(i, v + 10, str(v), ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(anomaly_dir, "1_price_and_stay_anomalies.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 2. DEMOGRAPHIC & GUEST COUNT ANOMALIES
# ==========================================
def plot_demographic_anomalies(df: pd.DataFrame, anomaly_dir: str):
    """
    Visualizes ghost bookings (0 guests), unaccompanied children (0 adults),
    and extreme guest count outliers (up to 55 adults / 10 babies).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        '2. Guest Demographic Anomalies (Ghost Bookings & Extreme Group Outliers)', 
        fontsize=14, fontweight='bold'
    )

    children_clean = df['children'].fillna(0)
    total_guests = df['adults'] + children_clean + df['babies']

    # Subplot 1: Ghost Bookings & Unaccompanied Minors
    ghost_count = (total_guests == 0).sum()
    no_adult_with_kids = ((df['adults'] == 0) & ((children_clean + df['babies']) > 0)).sum()
    single_Group_outliers = (df['adults'] >= 10).sum()

    anomaly_types = [
        'Ghost Bookings\n(Total Guests == 0)', 
        'Unaccompanied Kids\n(Adults == 0, Kids > 0)', 
        'Mass Group Rows\n(Adults >= 10 in 1 Row)'
    ]
    counts = [ghost_count, no_adult_with_kids, single_Group_outliers]
    
    sns.barplot(x=anomaly_types, y=counts, ax=axes[0], palette='Reds_r', hue=anomaly_types, legend=False)
    axes[0].set_title("Invalid / Suspicious Guest Combinations")
    axes[0].set_ylabel("Number of Rows")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

    # Subplot 2: 99th Percentile vs Maximum Value Comparison
    metrics = ['adults', 'children', 'babies', 'required_car_parking_spaces']
    q99_vals = [df[m].quantile(0.99) for m in metrics]
    max_vals = [df[m].max() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width/2, q99_vals, width, label='99th Percentile (Normal Max)', color='teal')
    axes[1].bar(x + width/2, max_vals, width, label='Absolute Max (Outlier)', color='crimson')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Adults', 'Children', 'Babies', 'Parking'])
    axes[1].set_title("99th Percentile vs. Extreme Max Outliers")
    axes[1].set_ylabel("Count per Booking")
    axes[1].legend()
    for i in range(len(metrics)):
        axes[1].text(x[i] - width/2, q99_vals[i] + 0.8, f"{q99_vals[i]:.0f}", ha='center', fontsize=9)
        axes[1].text(x[i] + width/2, max_vals[i] + 0.8, f"{max_vals[i]:.0f}", ha='center', fontweight='bold', color='crimson')

    # Subplot 3: Distribution of Extreme Adult Counts (> 4 Adults)
    extreme_adults = df[df['adults'] > 4]['adults'].value_counts().sort_index()
    sns.barplot(
        x=extreme_adults.index.astype(str), y=extreme_adults.values, 
        ax=axes[2], color='indianred'
    )
    axes[2].set_title(f"Bookings with > 4 Adults (Total: {extreme_adults.sum()} rows)")
    axes[2].set_xlabel("Number of Adults in Single Booking")
    axes[2].set_ylabel("Row Count")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(anomaly_dir, "2_demographic_anomalies.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 3. OPERATIONAL & BATCH CANCELLATION ANOMALIES
# ==========================================
def plot_operational_anomalies(df: pd.DataFrame, anomaly_dir: str):
    """
    Visualizes the Non-Refundable cancellation paradox (99.3% canceled),
    reservation_status_date batch spikes, and long waiting list outliers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        '3. Operational Anomalies: The Non-Refund Paradox & Batch Cancellation Spikes', 
        fontsize=14, fontweight='bold'
    )

    # Subplot 1: The Non-Refundable Cancellation Paradox
    ct_deposit = pd.crosstab(df['deposit_type'], df['is_canceled'], normalize='index') * 100
    ct_deposit.plot(
        kind='bar', stacked=True, ax=axes[0], 
        color=['seagreen', 'crimson'], rot=0
    )
    axes[0].set_title("The Deposit Paradox: 99.3% of 'Non Refund' Canceled!")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("Deposit Type")
    axes[0].legend(['Not Canceled (0)', 'Canceled (1)'], loc='lower right')

    # Subplot 2: Top 10 Anomaly Spikes in reservation_status_date
    status_spikes = df['reservation_status_date'].value_counts().head(10)
    sns.barplot(
        x=status_spikes.values, y=status_spikes.index, 
        ax=axes[1], palette='magma', hue=status_spikes.index, legend=False
    )
    axes[1].axvline(df['reservation_status_date'].value_counts().median(), color='cyan', linestyle='--', linewidth=2, label='Daily Median (115)')
    axes[1].set_title("Top 10 Single-Day Batch Spikes (reservation_status_date)")
    axes[1].set_xlabel("Bookings Updated on Single Date")
    axes[1].set_ylabel("Date")
    axes[1].legend()

    # Subplot 3: Lead Time Extreme Tail (> 365 Days)
    over_1yr = df[df['lead_time'] > 365]
    sns.histplot(over_1yr['lead_time'], bins=30, ax=axes[2], color='darkorchid')
    axes[2].set_title(f"Bookings Made > 1 Year in Advance ({len(over_1yr)} rows, Max: 737d)")
    axes[2].set_xlabel("Lead Time (Days)")
    axes[2].set_ylabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(anomaly_dir, "3_operational_and_batch_anomalies.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 4. MISSING, DUPLICATES & 'UNDEFINED' SUMMARY
# ==========================================
def plot_data_quality_summary(df: pd.DataFrame, anomaly_dir: str):
    """
    Visualizes missing value percentages, exact duplicate row ratios,
    and hidden 'Undefined' string categories across columns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        '4. Data Quality Overview: Duplicates, Missing Values & Hidden "Undefined" Labels', 
        fontsize=14, fontweight='bold'
    )

    # Subplot 1: Exact Duplicate Rows vs Unique Rows
    n_dups = df.duplicated().sum()
    n_unique = len(df) - n_dups
    sns.barplot(
        x=['Unique Rows', 'Exact Duplicate Rows'], 
        y=[n_unique, n_dups], ax=axes[0], 
        palette=['steelblue', 'crimson'], hue=['Unique Rows', 'Exact Duplicate Rows'], legend=False
    )
    axes[0].set_title(f"Duplicate Rows Anomaly ({n_dups / len(df) * 100:.1f}% of Dataset!)")
    axes[0].set_ylabel("Number of Rows")
    for i, v in enumerate([n_unique, n_dups]):
        axes[0].text(i, v + 1000, f"{v:,}", ha='center', fontweight='bold')

    # Subplot 2: Missing Values (NaN) Count (Log Scale for visibility of children/country)
    missing_cols = df[['company', 'agent', 'country', 'children']].isnull().sum()
    sns.barplot(
        x=missing_cols.index, y=missing_cols.values, 
        ax=axes[1], palette='Oranges_r', hue=missing_cols.index, legend=False
    )
    axes[1].set_yscale('log')
    axes[1].set_title("Missing Values Count per Column (Log Scale)")
    axes[1].set_ylabel("Missing Count (Log Scale)")
    for i, v in enumerate(missing_cols.values):
        axes[1].text(i, v * 1.2, f"{v:,}\n({v/len(df)*100:.2f}%)", ha='center', fontsize=9, fontweight='bold')

    # Subplot 3: Hidden 'Undefined' String Values in Categorical Columns
    undef_counts = {
        'meal': (df['meal'] == 'Undefined').sum(),
        'distribution_channel': (df['distribution_channel'] == 'Undefined').sum(),
        'market_segment': (df['market_segment'] == 'Undefined').sum()
    }
    sns.barplot(
        x=list(undef_counts.keys()), y=list(undef_counts.values()), 
        ax=axes[2], palette='Purples_r', hue=list(undef_counts.keys()), legend=False
    )
    axes[2].set_title("Hidden 'Undefined' String Categories")
    axes[2].set_ylabel("Row Count")
    for i, v in enumerate(undef_counts.values()):
        axes[2].text(i, v + 15, str(v), ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(anomaly_dir, "4_missing_and_undefined_summary.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    # Set visual style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.dpi'] = 120

    # Load raw dataset
    input_path = "../dataset/hotel_bookings.csv"
    df = pd.read_csv(input_path, na_values=['NULL', 'NA'], keep_default_na=True)

    # Create separate output directory for anomaly plots
    anomaly_dir = "./anomaly_plots"
    os.makedirs(anomaly_dir, exist_ok=True)

    print("Generating Data Anomaly plots in separate folder...")
    plot_price_and_stay_anomalies(df, anomaly_dir)
    plot_demographic_anomalies(df, anomaly_dir)
    plot_operational_anomalies(df, anomaly_dir)
    plot_data_quality_summary(df, anomaly_dir)
    print(f"\nAll anomaly plots successfully saved in: {os.path.abspath(anomaly_dir)}")