"""Daily per-epoch activity-level features from acc/gyro/GPS datasets."""
import pandas as pd
import numpy as np
from datetime import time

# (Optional) shapely for polygon containment; falls back to bounding box if unavailable
try:
    from shapely.geometry import Point, Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False
    Point = Polygon = None


def assign_epoch(dt):
    t = dt.time()
    if time(0, 0) <= t < time(9, 0):
        return 1  # Morning
    elif time(9, 0) <= t < time(18, 0):
        return 2  # Day
    elif time(18, 0) <= t <= time(23, 59, 59):
        return 3  # Night


def load_and_prepare(filepath, timestamp_col='timestamp'):
    df = pd.read_csv(filepath)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', utc=True)
    df.dropna(subset=[timestamp_col], inplace=True)
    df['local_time'] = df[timestamp_col].dt.tz_convert('America/New_York')
    df['date'] = df['local_time'].dt.date
    df['epoch'] = df['local_time'].apply(assign_epoch)
    return df


def add_full_day_epoch(df):
    full_day = df.copy()
    full_day['epoch'] = 0
    return pd.concat([df, full_day], ignore_index=True)


def extract_acc_features(df, sampling_rate=50.0):
    feats = []
    for (user, date, epoch), group in df.groupby(['user', 'date', 'epoch']):
        acc = group['acc_magnitude'].dropna().values

        # Durations in seconds (count / actual sampling rate, not a hardcoded 50 Hz)
        sedentary_dur = np.sum(acc < 1.3) / sampling_rate
        walking_dur = np.sum((acc >= 1.3) & (acc < 2.3)) / sampling_rate
        run_dur = np.sum(acc >= 2.3) / sampling_rate

        # Statistical features
        acc_mean = np.mean(acc)
        acc_var = np.var(acc)
        acc_max = np.max(acc)
        acc_min = np.min(acc)
        acc_std = np.std(acc)
        acc_energy = np.sum(acc**2)

        feats.append({
            'user': user,
            'date': date,
            'epoch': epoch,
            'acc_sedentary_min': sedentary_dur / 60,
            'acc_walking_min': walking_dur / 60,
            'acc_running_min': run_dur / 60,
            'acc_avg_magnitude': acc_mean,
            'acc_var': acc_var,
            'acc_max': acc_max,
            'acc_min': acc_min,
            'acc_std': acc_std,
            'acc_energy': acc_energy
        })
    return pd.DataFrame(feats)


def extract_gyro_features(df):
    feats = []
    for (user, date, epoch), group in df.groupby(['user', 'date', 'epoch']):
        g = group['gyro_magnitude'].dropna().values
        var_gyro = np.var(g)
        avg_gyro = np.mean(g)
        max_gyro = np.max(g)
        min_gyro = np.min(g)
        std_gyro = np.std(g)
        energy_gyro = np.sum(g**2)

        feats.append({
            'user': user,
            'date': date,
            'epoch': epoch,
            'gyro_avg_magnitude': avg_gyro,
            'gyro_var': var_gyro,
            'gyro_max': max_gyro,
            'gyro_min': min_gyro,
            'gyro_std': std_gyro,
            'gyro_energy': energy_gyro
        })
    return pd.DataFrame(feats)


UMBC_POLYGON = [
    (39.2565, -76.7115),
    (39.2550, -76.7150),
    (39.2520, -76.7158),
    (39.2495, -76.7155),
    (39.2482, -76.7135),
    (39.2475, -76.7098),
    (39.2486, -76.7055),
    (39.2505, -76.7048),
    (39.2525, -76.7042),
    (39.2542, -76.7050),
    (39.2560, -76.7072),
    (39.2565, -76.7115)
]

# Precompute polygon shape (Shapely) or bounding box fallback
if _HAS_SHAPELY:
    _UMBC_POLY_SHAPE = Polygon([(lon, lat) for (lat, lon) in UMBC_POLYGON])  # shapely expects (x=lon, y=lat)
else:
    _UMBC_LAT_MIN = min(lat for lat, _ in UMBC_POLYGON)
    _UMBC_LAT_MAX = max(lat for lat, _ in UMBC_POLYGON)
    _UMBC_LON_MIN = min(lon for _, lon in UMBC_POLYGON)
    _UMBC_LON_MAX = max(lon for _, lon in UMBC_POLYGON)


def is_on_campus_polygon(lat, lon):
    """Return True if (lat, lon) falls within UMBC polygon (Shapely if available, else bounding box)."""
    if _HAS_SHAPELY:
        return _UMBC_POLY_SHAPE.contains(Point(lon, lat))
    # bounding-box fallback (broader than polygon but preserves intent)
    return (_UMBC_LAT_MIN <= lat <= _UMBC_LAT_MAX) and (_UMBC_LON_MIN <= lon <= _UMBC_LON_MAX)


def extract_gps_features(df, sampling_rate=1):
    feats = []

    df = df.copy()
    df['on_campus'] = df.apply(
        lambda row: is_on_campus_polygon(row['latitude'], row['longitude']),
        axis=1
    )

    for (user, date, epoch), group in df.groupby(['user', 'date', 'epoch']):
        group = group.sort_values('timestamp')

        total_distance = group['distance'].sum()
        max_speed = group['speed'].max()
        avg_speed = group['speed'].mean()
        max_distance = group['distance'].max()
        speed_var = group['speed'].var()
        num_points = len(group)
        stop_duration = np.sum(group['speed'] < 0.2) / sampling_rate
        on_campus_time = group['on_campus'].sum() / sampling_rate

        feats.append({
            'user': user,
            'date': date,
            'epoch': epoch,
            'gps_total_dist_m': total_distance,
            'gps_max_speed': max_speed,
            'gps_avg_speed': avg_speed,
            'gps_max_dist_from_prev': max_distance,
            'gps_speed_var': speed_var,
            'gps_num_points': num_points,
            'gps_stop_duration_sec': stop_duration,
            'gps_on_campus_min': on_campus_time / 60
        })

    return pd.DataFrame(feats)


def extract_daily_activity_features(acc_path: str,
                                    gyro_path: str,
                                    gps_path: str,
                                    output_path: str) -> pd.DataFrame:
    """Load acc/gyro/GPS datasets, add a full-day epoch, extract per-epoch
    features, pivot to wide (feature_epochX columns), and save."""
    # Load
    acc_df = load_and_prepare(acc_path)
    gyro_df = load_and_prepare(gyro_path)
    gps_df  = load_and_prepare(gps_path)


    if 'magnitude' in acc_df.columns and 'acc_magnitude' not in acc_df.columns:
        acc_df['acc_magnitude'] = acc_df['magnitude']
    if 'magnitude' in gyro_df.columns and 'gyro_magnitude' not in gyro_df.columns:
        gyro_df['gyro_magnitude'] = gyro_df['magnitude']

    # Estimate the accelerometer sampling rate from timestamps (Beiwe is not 50 Hz).
    _dt = acc_df['local_time'].sort_values().diff().dt.total_seconds()
    _dt = _dt[_dt > 0]
    acc_rate = (1.0 / _dt.median()) if len(_dt) else 50.0

    # Add epoch 0
    acc_df = add_full_day_epoch(acc_df)
    gyro_df = add_full_day_epoch(gyro_df)
    gps_df  = add_full_day_epoch(gps_df)

    # Extract features
    acc_feats  = extract_acc_features(acc_df, sampling_rate=acc_rate)
    gyro_feats = extract_gyro_features(gyro_df)
    gps_feats  = extract_gps_features(gps_df)

    # Merge all feature tables
    combined = acc_feats.merge(gyro_feats, on=['user', 'date', 'epoch'], how='outer')
    combined = combined.merge(gps_feats, on=['user', 'date', 'epoch'], how='outer')

    # Pivot to wide
    wide = combined.pivot_table(index=['user', 'date'],
                                columns='epoch',
                                values=[c for c in combined.columns if c not in ['user', 'date', 'epoch']],
                                aggfunc='first')

    # Flatten MultiIndex columns: (feature, epoch) -> feature_epochX
    wide.columns = [f'{feat}_epoch{int(epoch)}' for feat, epoch in wide.columns]
    wide = wide.reset_index()

    # Save & return
    wide.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")
    return wide
