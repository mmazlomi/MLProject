import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_float_dtype, is_bool_dtype


def inspect_dataset_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive structural report of all columns in the dataset,
    separating categorical and numerical features for deep inspection.
    """
    report = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = df[col].isnull().sum()
        pct_missing = round((n_missing / len(df)) * 100, 3)
        n_unique = df[col].nunique()
        
        # Safely check if the column is numeric (excluding boolean)
        if is_numeric_dtype(df[col]) and not is_bool_dtype(df[col]):
            min_val = df[col].min()
            median_val = df[col].median()
            q99_val = round(df[col].quantile(0.99), 2)
            max_val = df[col].max()
            details = f"Min: {min_val} | Med: {median_val} | Q99: {q99_val} | Max: {max_val}"
        else:
            # Handle string, object, and categorical columns
            vc = df[col].value_counts().head(5)
            top_vals = [f"{k} ({v})" for k, v in vc.items()]
            details = f"Top: {', '.join(top_vals)}"
            
        report.append({
            'Column': col,
            'Dtype': dtype,
            'Missing_Cnt': n_missing,
            'Missing (%)': pct_missing,
            'Unique_Vals': n_unique,
            'Summary / Range': details
        })
        
    report_df = pd.DataFrame(report)
    return report_df


# ==========================================
# 1. CONTINUOUS & SKEWED COLUMNS (Hist + Boxplot)
# ==========================================
def plot_continuous_columns(df: pd.DataFrame, output_dir: str):
    """
    Plots Histogram + Boxplot for continuous numerical columns
    to inspect both distribution shape and extreme outliers.
    """
    cont_cols = ['lead_time', 'adr', 'days_in_waiting_list']
    fig, axes = plt.subplots(len(cont_cols), 2, figsize=(14, 10))
    fig.suptitle(
        '1. Continuous Features: Distribution (99th Percentile) vs. Full Boxplot (Outliers)', 
        fontsize=14, fontweight='bold'
    )

    for i, col in enumerate(cont_cols):
        # Left: Histogram capped at 99th percentile for readability
        q99 = df[col].quantile(0.99)
        subset = df[df[col] <= q99][col].dropna()
        sns.histplot(subset, bins=40, kde=True, ax=axes[i, 0], color='steelblue')
        axes[i, 0].set_title(f"{col} - Histogram (Capped at Q99 = {q99:.1f})")
        
        # Right: Full Boxplot to expose extreme outliers (e.g., adr = 5400 or -6.38)
        sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='salmon')
        axes[i, 1].set_title(f"{col} - Boxplot (Full Range including Outliers)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "1_continuous_distributions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 2. DISCRETE COUNT COLUMNS (Countplots)
# ==========================================
def plot_discrete_columns(df: pd.DataFrame, output_dir: str):
    """
    Plots bar charts of value counts for discrete integer columns.
    """
    discrete_cols = [
        'stays_in_weekend_nights', 'stays_in_week_nights', 
        'adults', 'children', 'babies', 
        'booking_changes', 'required_car_parking_spaces', 
        'total_of_special_requests', 'previous_cancellations',
        'previous_bookings_not_canceled'
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()
    fig.suptitle(
        '2. Discrete Count Features (Top 10 Most Frequent Values per Column)', 
        fontsize=14, fontweight='bold'
    )

    for i, col in enumerate(discrete_cols):
        # Take top 10 most frequent discrete values to avoid long empty tails
        vc = df[col].dropna().astype(int).value_counts().head(10).sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=axes[i], color='teal')
        axes[i].set_title(f"{col} (Max: {df[col].max():.0f})")
        axes[i].set_ylabel("Count")
        axes[i].set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "2_discrete_counts.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 3. CATEGORICAL COLUMNS (Low & High Cardinality)
# ==========================================
def plot_categorical_columns(df: pd.DataFrame, output_dir: str):
    """
    Plots horizontal bar charts for categorical and binary columns.
    Safely handles float ID columns (agent, company) without triggering Pandas4Warning.
    """
    cat_cols = [
        'hotel', 'is_canceled', 'is_repeated_guest', 'meal', 
        'market_segment', 'distribution_channel', 'deposit_type', 
        'customer_type', 'reservation_status', 'reserved_room_type', 
        'assigned_room_type', 'country', 'agent', 'company'
    ]
    
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()
    fig.suptitle(
        '3. Categorical, Binary & ID Features (Top 10 Categories for High-Cardinality Columns)', 
        fontsize=14, fontweight='bold'
    )

    for i, col in enumerate(cat_cols):
        # Type-safe handling for float columns with NaNs (avoids Pandas4Warning & '.0' string bug)
        if is_float_dtype(df[col]):
            series = df[col].dropna().astype(int).astype(str).reindex(df.index, fill_value='Missing')
        else:
            series = df[col].astype(object).fillna('Missing').astype(str)
        
        vc = series.value_counts().head(10)
        sns.barplot(
            x=vc.values, y=vc.index, ax=axes[i], 
            palette="viridis", hue=vc.index, legend=False
        )
        
        n_unique = df[col].nunique()
        title_suffix = f"(Top 10 of {n_unique})" if n_unique > 10 else f"({n_unique} unique)"
        axes[i].set_title(f"{col} {title_suffix}")
        axes[i].set_xlabel("Bookings Count")
        axes[i].set_ylabel("")

    # Hide the 15th empty subplot slot (since we have 14 columns in a 5x3 grid)
    axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, "3_categorical_distributions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================
# 4. TEMPORAL / DATE COLUMNS (Time-Series & Seasonality)
# ==========================================
def plot_temporal_columns(df: pd.DataFrame, output_dir: str):
    """
    Plots chronological trends for year, ordered month, week number, day of month,
    and reservation_status_date.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(
        '4. Temporal Features: Seasonality & Arrival Patterns', 
        fontsize=14, fontweight='bold'
    )

    # 1. Arrival Year
    sns.countplot(data=df, x='arrival_date_year', hue='hotel', ax=axes[0])
    axes[0].set_title("Bookings by Year and Hotel Type")

    # 2. Arrival Month (Ordered chronologically)
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_counts = df['arrival_date_month'].value_counts().reindex(month_order)
    sns.lineplot(
        x=month_counts.index, y=month_counts.values, 
        marker='o', linewidth=2.5, ax=axes[1], color='crimson'
    )
    axes[1].set_title("Monthly Seasonality (Ordered Jan - Dec)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel("Total Bookings")

    # 3. Arrival Week Number (1 to 53)
    week_counts = df['arrival_date_week_number'].value_counts().sort_index()
    sns.lineplot(x=week_counts.index, y=week_counts.values, ax=axes[2], color='darkorange')
    axes[2].set_title("Bookings across Week of the Year (Weeks 1 - 53)")
    axes[2].set_xlabel("Week Number")

    # 4. Arrival Day of Month (1 to 31)
    day_counts = df['arrival_date_day_of_month'].value_counts().sort_index()
    sns.barplot(x=day_counts.index, y=day_counts.values, ax=axes[3], color='royalblue')
    axes[3].set_title("Bookings by Day of the Month (Days 1 - 31)")
    axes[3].set_xlabel("Day of Month")
    axes[3].tick_params(axis='x', labelsize=8)

    # 5. Reservation Status Date (Monthly aggregated time series)
    status_dates = pd.to_datetime(df['reservation_status_date']).dt.to_period('M').value_counts().sort_index()
    status_dates.index = status_dates.index.astype(str)
    sns.lineplot(x=status_dates.index, y=status_dates.values, marker='o', ax=axes[4], color='purple')
    axes[4].set_title("Reservation Status Updates (Monthly Trend)")
    axes[4].tick_params(axis='x', rotation=45, labelsize=8)
    axes[4].set_xlabel("Year-Month")

    # Hide the 6th empty subplot slot
    axes[5].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "4_temporal_patterns.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    # Configure pandas display options for terminal output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    input_path = "../dataset/hotel_bookings.csv"
    df = pd.read_csv(input_path, na_values=['NULL', 'NA'], keep_default_na=True)

    # Generate and print the structural report in terminal
    report_df = inspect_dataset_fields(df)
    print(report_df.to_string(index=False))

    # Set visual style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.dpi'] = 120

    # Create output directory for plots
    output_dir = "./eda_plots"
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating EDA plots for all columns...")
    plot_continuous_columns(df, output_dir)
    plot_discrete_columns(df, output_dir)
    plot_categorical_columns(df, output_dir)
    plot_temporal_columns(df, output_dir)
    print(f"\nAll plots successfully saved in: {os.path.abspath(output_dir)}")