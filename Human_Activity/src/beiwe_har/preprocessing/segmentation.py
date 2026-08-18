"""Session-aware segmentation with resampling and optional gravity removal.

Beiwe accelerometer is sampled irregularly (and slower than the 50 Hz the model
was trained on), so each continuous session is resampled onto a uniform
`frequency` grid before being cut into fixed windows. `remove_gravity=True`
subtracts a low-pass gravity estimate per axis, yielding gravity-free "user
acceleration" — this matches the MotionSense signal the RF/ECDF model was
trained on.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt


def _low_pass(sig, fs, cutoff=0.3, order=3):
    """Gravity estimate: low-pass each axis; mean fallback for short sessions."""
    if len(sig) <= 3 * (order + 1):
        return np.full(len(sig), float(np.mean(sig)))
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, sig)


def _resample_session(session_df, axis_cols, frequency):
    """Resample one gap-free session onto a uniform `frequency` Hz time grid."""
    ax = list(axis_cols)
    s = session_df.set_index("timestamp")[ax].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if len(s) < 2:
        return None
    period_ms = int(round(1000.0 / frequency))
    grid = pd.date_range(s.index[0], s.index[-1], freq=f"{period_ms}ms")
    if len(grid) < 2:
        return None
    return s.reindex(s.index.union(grid)).interpolate(method="time").reindex(grid)


def generate_windowed_data(
    file_path,
    *,
    frequency: int = 50,
    window_size_in_seconds: float = 2.0,
    time_gap_threshold: float = 10.0,     # seconds; a larger gap starts a new session
    overlap_ratio: float = 0.5,
    remove_gravity: bool = False,         # subtract low-pass gravity -> user acceleration
    axis_cols=("x", "y", "z"),
    output_path=None,
) -> pd.DataFrame:
    """Segment a combined dataset (user, timestamp, x, y, z) into fixed windows.

    Emits columns x_0..x_{N-1}, y_*, z_* per window (N = frequency * window_sec)
    plus user, window_start_time, session_id. Writes `<stem>_segmented.csv`.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    req = {"user", "timestamp", *axis_cols}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).dropna(subset=list(axis_cols))
    df = df.sort_values(["user", "timestamp"], kind="mergesort").reset_index(drop=True)

    dt = df.groupby("user", sort=False)["timestamp"].diff().dt.total_seconds()
    df["session_id"] = ((dt.isna()) | (dt > float(time_gap_threshold))).cumsum()

    window_size = max(1, int(round(window_size_in_seconds * int(frequency))))
    step_size = min(max(1, int(round(window_size * (1.0 - float(overlap_ratio))))), window_size)
    ax_x, ax_y, ax_z = axis_cols

    rows = []
    for (user, session_id), session_df in df.groupby(["user", "session_id"], sort=False):
        grid = _resample_session(session_df, axis_cols, frequency)
        if grid is None or len(grid) < window_size:
            continue
        if remove_gravity:
            for ax in axis_cols:
                grid[ax] = grid[ax].to_numpy() - _low_pass(grid[ax].to_numpy(), frequency)
        x, y, z = grid[ax_x].to_numpy(), grid[ax_y].to_numpy(), grid[ax_z].to_numpy()
        times = grid.index
        for start in range(0, len(grid) - window_size + 1, step_size):
            row = {"user": user, "window_start_time": times[start], "session_id": int(session_id)}
            for i in range(window_size):
                row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"] = x[start + i], y[start + i], z[start + i]
            rows.append(row)

    acc_columns = [f"{axis}_{i}" for i in range(window_size) for axis in ("x", "y", "z")]
    final_columns = ["user", "window_start_time", "session_id"] + acc_columns
    df_segmented = pd.DataFrame(rows).reindex(columns=final_columns)

    if output_path is None:
        output_path = file_path.with_name(file_path.stem + "_segmented.csv")
    df_segmented.to_csv(output_path, index=False)
    print(f"Segmented file saved to: {output_path}  ({len(df_segmented)} windows)")
    return df_segmented
