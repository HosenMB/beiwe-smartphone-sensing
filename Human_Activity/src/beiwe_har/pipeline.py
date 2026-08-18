"""End-to-end HAR pipeline: raw Beiwe study folder -> features, predictions, summaries.

    from beiwe_har import run_pipeline
    run_pipeline("Beiwe_Data")

Steps: build per-stream datasets -> resample + (gravity-removed) segment
accelerometer -> ECDF features -> activity predictions -> per-minute/day
summaries. UCI-HAR features, and the gyro/GPS-dependent daily activity-level
table, are produced only when requested / when those streams exist.
"""
from pathlib import Path

from . import config
from .preprocessing.dataset import (
    create_accelerometer_dataset, create_gyroscope_dataset, create_gps_dataset,
    ACC_OUT, GYRO_OUT, GPS_OUT,
)
from .preprocessing.segmentation import generate_windowed_data
from .features.ecdf import extract_ECDF_features
from .features.uci_har import extract_UCI_HAR_features
from .features.activity_level import extract_daily_activity_features
from .models.har_model import predict_har_from_ecdf
from .outputs.summary import create_summary


def run_pipeline(study_dir, output_dir=None, frequency=config.SAMPLE_RATE_HZ,
                 window_sec=config.WINDOW_SEC, overlap=0.5, uci_har=False):
    """Run the full pipeline on a Beiwe study folder and return a dict of outputs."""
    study = Path(study_dir)
    dest = Path(output_dir) if output_dir else study
    dest.mkdir(parents=True, exist_ok=True)
    freq = int(frequency)
    out = {}

    # 1) Combine raw per-stream files into `dest` (gyro/GPS skip silently if absent).
    create_accelerometer_dataset(str(study), out_dir=str(dest))
    acc_ds = dest / ACC_OUT
    if not acc_ds.exists():
        raise RuntimeError(f"No accelerometer data found under {study}.")
    create_gyroscope_dataset(str(study), out_dir=str(dest))
    create_gps_dataset(str(study), out_dir=str(dest))
    gyro_ds, gps_ds = dest / GYRO_OUT, dest / GPS_OUT
    out["acc_dataset"] = acc_ds

    # 2) Resample + segment as GRAVITY-REMOVED user acceleration (matches the model).
    seg = acc_ds.with_name(acc_ds.stem + "_segmented.csv")
    generate_windowed_data(acc_ds, frequency=freq, window_size_in_seconds=window_sec,
                           overlap_ratio=overlap, remove_gravity=True, output_path=seg)
    out["acc_segmented"] = seg

    # 3) ECDF features -> activity predictions.
    extract_ECDF_features(str(seg), frequency=freq, duration_in_seconds=window_sec)
    ecdf_csv = seg.with_name(seg.stem.replace("_segmented", "") + "_ECDF_features.csv")
    predict_har_from_ecdf(str(ecdf_csv))
    out["ecdf_features"] = ecdf_csv
    out["predictions"] = ecdf_csv.with_name(ecdf_csv.stem.replace("_ECDF_features", "") + "_predictions.csv")

    # 4) Optional UCI-HAR features (needs RAW signal; it removes gravity internally).
    if uci_har:
        raw_seg = acc_ds.with_name(acc_ds.stem + "_raw_segmented.csv")
        generate_windowed_data(acc_ds, frequency=freq, window_size_in_seconds=window_sec,
                               overlap_ratio=overlap, remove_gravity=False, output_path=raw_seg)
        extract_UCI_HAR_features(str(raw_seg), frequency=freq, duration_in_seconds=window_sec)
        out["uci_har_features"] = raw_seg.with_name(raw_seg.stem.replace("_segmented", "_UCI_HAR_features") + ".csv")

    # 5) Per-minute + day-wise activity summaries (accelerometer only).
    create_summary(str(acc_ds))
    out["per_minute_summary"] = acc_ds.with_name("acc_aggregated.csv")
    out["daily_summary"] = acc_ds.with_name("daily_summary.csv")

    # 6) Daily activity-level features (needs acc + gyro + GPS).
    if gyro_ds.exists() and gps_ds.exists():
        dal = dest / "daily_activity_features.csv"
        extract_daily_activity_features(str(acc_ds), str(gyro_ds), str(gps_ds), str(dal))
        out["daily_activity_features"] = dal

    print("\nPipeline complete. Outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="beiwe-har", description="Beiwe study -> HAR features, predictions, summaries.")
    p.add_argument("study_dir", help="Folder with <participant>/accelerometer/*.csv")
    p.add_argument("-o", "--output-dir", default=None, help="where to write outputs (default: study folder)")
    p.add_argument("--frequency", type=float, default=config.SAMPLE_RATE_HZ)
    p.add_argument("--window-sec", type=float, default=config.WINDOW_SEC)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--uci-har", action="store_true", help="also compute UCI-HAR features")
    a = p.parse_args(argv)
    run_pipeline(a.study_dir, output_dir=a.output_dir, frequency=a.frequency,
                 window_sec=a.window_sec, overlap=a.overlap, uci_har=a.uci_har)
