
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def summarize_accelerometer_by_minute(
    input_file: str,
    threshold_low: float = 1.3,
    threshold_high: float = 2.3,
) -> pd.DataFrame:
    """
    Aggregates accelerometer data by user and minute, and saves a summary with:
    - Mean and Std of acc_magnitude
    - Activity_Level (Low / Medium / High) based on thresholds

    Assumes input CSV has columns: user, timestamp, acc_x, acc_y, acc_z, acc_magnitude
    and that 'timestamp' is timezone-aware.
    """
    accel_df = pd.read_csv(input_file)
    accel_df['timestamp'] = pd.to_datetime(accel_df['timestamp'], format='mixed')


    if 'acc_magnitude' not in accel_df.columns and 'magnitude' in accel_df.columns:
        accel_df['acc_magnitude'] = accel_df['magnitude']

    accel_df.set_index('timestamp', inplace=True)

    if accel_df.index.tz is None:
        raise ValueError("timestamp column must be timezone-aware (e.g., US/Eastern).")

    summary = (
        accel_df
        .groupby('user')['acc_magnitude']
        .resample('1min')
        .agg(['mean', 'std'])
        .dropna()
        .rename(columns={'mean': 'Mean_Mag', 'std': 'Std_Mag'})
        .reset_index()
    )

    def classify_activity(val_g: float) -> str:
        if val_g < threshold_low:
            return "Low"
        elif val_g < threshold_high:
            return "Medium"
        else:
            return "High"

    summary['Activity_Level'] = summary['Mean_Mag'].apply(classify_activity)

    # Save next to input with default name
    output_path = Path(input_file).with_name("acc_aggregated.csv")
    summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved activity summary to {output_path}")

    return summary


def summarize_daily_activity(
    input_file: str,
    threshold_low: float = 1.3,
    threshold_high: float = 2.3,
) -> pd.DataFrame:
    """
    Creates a day-wise summary from accelerometer data and saves it next to input as 'daily_summary.csv'.
    """
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')


    if 'acc_magnitude' not in df.columns and 'magnitude' in df.columns:
        df['acc_magnitude'] = df['magnitude']

    if df['timestamp'].dt.tz is None:
        raise ValueError("Timestamp column must be timezone-aware (e.g., US/Eastern).")

    df.set_index('timestamp', inplace=True)

    # Compute magnitude change (jerk approximation)
    df['jerk'] = df['acc_magnitude'].diff().abs()

    # Per-minute aggregates
    minute_df = (
        df.groupby('user')['acc_magnitude']
        .resample('1min')
        .agg(['mean', 'std', 'count'])
        .dropna()
        .rename(columns={'mean': 'Mean_Mag', 'std': 'Std_Mag', 'count': 'Samples'})
        .reset_index()
    )

    def classify_activity(mag: float) -> str:
        if mag < threshold_low:
            return 'Low'
        elif mag < threshold_high:
            return 'Medium'
        else:
            return 'High'

    minute_df['Activity_Level'] = minute_df['Mean_Mag'].apply(classify_activity)
    minute_df['date'] = minute_df['timestamp'].dt.date

    # Daily summary
    daily_summary = []
    for (user, date), group in minute_df.groupby(['user', 'date']):
        total_bursts = len(group)
        motion_minutes = group[group['Activity_Level'] != 'Low'].shape[0]
        sedentary_percent = 100 * group[group['Activity_Level'] == 'Low'].shape[0] / total_bursts
        avg_level = Counter(group['Activity_Level']).most_common(1)[0][0]

        jerk_max = df[df.index.date == date].groupby('user')['jerk'].max().get(user, np.nan)

        summary_row = {
            'user': user,
            'Date': date,
            'Total_Bursts': total_bursts,
            'Total_Motion_Time': f"{motion_minutes // 60} hr {motion_minutes % 60} min",
            'Sedentary_%': round(sedentary_percent, 1),
            'Max_Jerk': round(jerk_max, 3) if pd.notna(jerk_max) else np.nan,
            'Avg_Activity_Level': avg_level
        }
        daily_summary.append(summary_row)

    summary_df = pd.DataFrame(daily_summary)

    # Save next to input with default name
    output_path = Path(input_file).with_name("daily_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Daily summary saved to {output_path}")

    return summary_df


def create_summary(
    input_file: str,
    threshold_low: float = 1.3,
    threshold_high: float = 2.3,
):
    """
    Create both per-minute and day-wise summaries from an accelerometer dataset CSV.
    Saves:
      - acc_aggregated.csv
      - daily_summary.csv
    next to the input file.

    Returns:
        (perminute_df, daywise_df)
    """
    perminute_df = summarize_accelerometer_by_minute(
        input_file,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )
    daywise_df = summarize_daily_activity(
        input_file,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )
    return perminute_df, daywise_df
