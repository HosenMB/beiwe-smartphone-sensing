"""Combine a study's raw per-stream Beiwe CSVs into one dataset per stream.

Each `<study>/<user>/<stream>/*.csv` is read, its UTC time converted to
US/Eastern, and all users concatenated into `<study>/<stream>_dataset.csv`.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pytz
from geopy.distance import geodesic

ACC_OUT = "acc_dataset.csv"
GYRO_OUT = "gyro_dataset.csv"
GPS_OUT = "gps_dataset.csv"

_EASTERN = pytz.timezone("US/Eastern")


def _create_imu_dataset(study_path, stream, out_name, out_dir=None):
    """Accelerometer/gyroscope share the same tri-axial layout; build either one."""
    base = Path(study_path)
    rows = []
    for user_dir in base.glob("*"):
        stream_dir = user_dir / stream
        if not stream_dir.is_dir():
            continue
        for csv_file in stream_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, usecols=["UTC time", "x", "y", "z"])
                df["timestamp"] = pd.to_datetime(df["UTC time"]).dt.tz_localize("UTC").dt.tz_convert(_EASTERN)
                df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
                df["user"] = user_dir.name
                rows.append(df[["user", "timestamp", "x", "y", "z", "magnitude"]])
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    if not rows:
        print(f"No valid {stream} data found.")
        return
    out_path = Path(out_dir or base) / out_name
    pd.concat(rows, ignore_index=True).sort_values(["user", "timestamp"]).to_csv(out_path, index=False)
    print(f"Saved {stream} dataset to: {out_path}")


def create_accelerometer_dataset(study_path, out_dir=None):
    _create_imu_dataset(study_path, "accelerometer", ACC_OUT, out_dir)


def create_gyroscope_dataset(study_path, out_dir=None):
    _create_imu_dataset(study_path, "gyro", GYRO_OUT, out_dir)


def create_gps_dataset(study_path, out_dir=None):
    """Combine users' GPS CSVs, adding per-point distance and speed."""
    base = Path(study_path)
    rows = []
    for user_dir in base.glob("*"):
        gps_dir = user_dir / "gps"
        if not gps_dir.is_dir():
            continue
        for csv_file in gps_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, usecols=["UTC time", "latitude", "longitude", "altitude", "accuracy"])
                df["timestamp"] = pd.to_datetime(df["UTC time"]).dt.tz_localize("UTC").dt.tz_convert(_EASTERN)
                df = df.drop(columns=["UTC time"]).sort_values("timestamp")

                coords = list(zip(df["latitude"], df["longitude"]))
                df["distance"] = [0.0] + [geodesic(coords[i - 1], coords[i]).meters for i in range(1, len(coords))]
                df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
                df["speed"] = np.where(df["time_diff"] > 0, df["distance"] / df["time_diff"], 0.0)
                df["user"] = user_dir.name
                rows.append(df[["user", "timestamp", "latitude", "longitude", "altitude", "accuracy", "distance", "speed"]])
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    if not rows:
        print("No valid GPS data found.")
        return
    out_path = Path(out_dir or base) / GPS_OUT
    pd.concat(rows, ignore_index=True).sort_values(["user", "timestamp"]).to_csv(out_path, index=False)
    print(f"Saved GPS dataset to: {out_path}")


def create_dataset(study_path, out_dir=None):
    """Build acc/gyro/GPS datasets for a study."""
    create_accelerometer_dataset(study_path, out_dir)
    create_gyroscope_dataset(study_path, out_dir)
    create_gps_dataset(study_path, out_dir)
