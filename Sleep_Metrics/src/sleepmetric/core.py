"""End-to-end pipeline: raw phone accelerometer CSV -> interpretable nightly
sleep metrics, in Maryland local time. Rule-based, no wearable, no sleep stages.
Public entry points: analyze(), analyze_night(), analyze_beiwe(); everything
else is internal. A night is the noon-to-noon local window labelled by its
evening date (9pm Aug 18 - 7am Aug 19 -> "2024-08-18").
"""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ALGO_VERSION = "0.1.0"
TIMEZONE = "America/New_York"   # Maryland — nights are defined on this local clock

OUTPUT_COLUMNS = [
    "student_id", "date", "sleep_onset", "sleep_offset", "window_len_min",
    "total_sleep_min", "longest_still_bout_min", "num_awakenings_est",
    "waso_total_min", "sleep_efficiency", "sleep_midpoint",
    "data_coverage", "quality_ok", "algo_version", "params_hash",
]


@dataclass
class Config:
    """Tunable thresholds. Defaults suit a phone on the mattress; every metric
    is computable without changing any of them. params_hash fingerprints the
    settings so each output row is traceable to the config that produced it."""

    epoch_sec: float = 30.0
    still_enmo_mg: float = 30.0       # ENMO below this (milli-g) is "still"
    tilt_thresh_deg: float = 10.0     # gravity-vector change below this is "still"
    block_min: float = 5.0            # stillness must persist this long
    bridge_min: float = 30.0          # motion/missing gaps up to this stay in-window
    min_window_min: float = 180.0     # shorter windows are flagged low quality
    smooth_epochs: int = 5            # majority-smooth to damp single-epoch flips
    awake_burst_min: float = 5.0      # an awakening is an awake run at least this long
    min_coverage: float = 0.75        # quality_ok gate

    def params_hash(self) -> str:
        return hashlib.md5(repr(sorted(asdict(self).items())).encode()).hexdigest()[:8]


def _runs(mask):
    """Yield (start, end_inclusive) index ranges where mask is True."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


# --- ingest -----------------------------------------------------------------
_ACC_ALIASES = {
    "x": ["acc_x", "ax", "x", "accelerationx", "accelerometerx", "accel_x"],
    "y": ["acc_y", "ay", "y", "accelerationy", "accelerometery", "accel_y"],
    "z": ["acc_z", "az", "z", "accelerationz", "accelerometerz", "accel_z"],
}
_TIME_ALIASES = ["timestamp", "time", "datetime", "date_time", "utctime", "ts", "t"]


def _find_col(aliases, colmap):
    return next((colmap[a] for a in aliases if a in colmap), None)


def _parse_time(series):
    """Parse a timestamp column of ISO strings or Unix epochs (auto-detecting
    seconds / milliseconds / microseconds / nanoseconds)."""
    if pd.api.types.is_numeric_dtype(series):
        v = pd.to_numeric(series, errors="coerce")
        mx = float(v.abs().max())
        unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us" if mx < 1e17 else "ns"
        return pd.to_datetime(v, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _to_local(t):
    """Beiwe records timestamps in UTC. Convert to Maryland local time (tz-naive)
    so nights split on the participant's real clock and DST is handled correctly."""
    t = pd.to_datetime(t)
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")          # epoch/UTC input -> mark as UTC
    return t.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


def _to_gravity_units(frame):
    """Convert x/y/z to g in place if they look like m/s^2 (~9.8 at rest)."""
    if np.sqrt(frame.x**2 + frame.y**2 + frame.z**2).median() > 5.0:
        frame[["x", "y", "z"]] /= 9.81
    return frame


def _load(source, column_map=None):
    """Load a phone CSV (path or DataFrame) into a tidy frame (time, x, y, z in
    g) and infer the student id. Column names, units and epoch timestamps are
    auto-detected."""
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    colmap = {c.lower().replace(" ", ""): c for c in df.columns}

    if column_map:
        cols = {k: column_map.get(k) for k in ("timestamp", "acc_x", "acc_y", "acc_z")}
    else:
        cols = {"timestamp": _find_col(_TIME_ALIASES, colmap),
                "acc_x": _find_col(_ACC_ALIASES["x"], colmap),
                "acc_y": _find_col(_ACC_ALIASES["y"], colmap),
                "acc_z": _find_col(_ACC_ALIASES["z"], colmap)}

    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(
            f"Couldn't find column(s) {missing}. Found: {list(df.columns)}. "
            f"Pass column_map={{'timestamp': ..., 'acc_x': ..., 'acc_y': ..., "
            f"'acc_z': ...}} to name them explicitly."
        )

    out = pd.DataFrame({
        "time": _parse_time(df[cols["timestamp"]]),
        "x": pd.to_numeric(df[cols["acc_x"]], errors="coerce"),
        "y": pd.to_numeric(df[cols["acc_y"]], errors="coerce"),
        "z": pd.to_numeric(df[cols["acc_z"]], errors="coerce"),
    }).dropna().sort_values("time").reset_index(drop=True)

    if len(out) < 2:
        raise ValueError("Fewer than 2 valid samples after parsing — check the file.")
    _to_gravity_units(out)

    sid = None
    if "student_id" in colmap:
        vals = df[colmap["student_id"]].dropna().unique()
        sid = str(vals[0]) if len(vals) else None
    if sid is None:
        sid = ("dataset" if isinstance(source, pd.DataFrame)
               else os.path.splitext(os.path.basename(str(source)))[0])
    return out, sid


# --- Beiwe ingestion --------------------------------------------------------
def _beiwe_student_id(source):
    parts = os.path.normpath(str(source)).split(os.sep)
    if "accelerometer" in parts:
        i = parts.index("accelerometer")
        return parts[i - 1] if i > 0 else "beiwe"
    return os.path.splitext(os.path.basename(os.path.normpath(str(source))))[0] or "beiwe"


def load_beiwe(source):
    """Read a Beiwe accelerometer export (a folder of hourly CSVs, a glob, or a
    single CSV) and concatenate it into one raw DataFrame."""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "**", "*.csv"), recursive=True))
    else:
        files = sorted(glob.glob(str(source)))
    if not files:
        raise ValueError(f"No CSV files found at {source}.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def preprocess_beiwe(source, out_csv=None, student_id=None):
    """Turn a raw Beiwe export into a tidy per-sample frame (timestamp, acc_x,
    acc_y, acc_z in g, student_id). Saves to out_csv if given; returns the frame."""
    raw = load_beiwe(source)
    colmap = {c.lower().replace(" ", ""): c for c in raw.columns}
    tcol = _find_col(_TIME_ALIASES, colmap)
    acc = {k: _find_col(_ACC_ALIASES[k[-1]], colmap) for k in ("acc_x", "acc_y", "acc_z")}
    if tcol is None or None in acc.values():
        raise ValueError(f"Unexpected Beiwe columns: {list(raw.columns)}")

    tidy = pd.DataFrame({
        "time": _to_local(_parse_time(raw[tcol])),
        "x": pd.to_numeric(raw[acc["acc_x"]], errors="coerce"),
        "y": pd.to_numeric(raw[acc["acc_y"]], errors="coerce"),
        "z": pd.to_numeric(raw[acc["acc_z"]], errors="coerce"),
    }).dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    _to_gravity_units(tidy)

    out = pd.DataFrame({
        "timestamp": tidy["time"], "acc_x": tidy["x"].round(5),
        "acc_y": tidy["y"].round(5), "acc_z": tidy["z"].round(5),
        "student_id": student_id or _beiwe_student_id(source),
    })
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


def analyze_beiwe(source, save_preprocessed=None, config=None, student_id=None):
    """One-call Beiwe pipeline: concatenate the hourly files, preprocess (parse
    epoch-ms timestamps, normalise units), optionally save the cleaned CSV, then
    analyze. Returns one metrics row per night."""
    tidy = preprocess_beiwe(source, out_csv=save_preprocessed, student_id=student_id)
    return analyze(tidy, config=config)


# --- features (per epoch) ---------------------------------------------------
def _epoch_features(raw, cfg):
    """Per-epoch frame: time, enmo (mg), tilt_change (deg), coverage (0-1).
    Epochs with no samples become gaps (NaN activity, zero coverage)."""
    t0 = raw["time"].iloc[0]
    secs = (raw["time"] - t0).dt.total_seconds().to_numpy()
    ep = np.floor(secs / cfg.epoch_sec).astype(int)
    enmo = np.clip(np.sqrt(raw.x**2 + raw.y**2 + raw.z**2) - 1.0, 0, None) * 1000.0

    dt = np.median(np.diff(secs))
    expected = max(cfg.epoch_sec / dt if dt > 0 else 1.0, 1.0)

    agg = pd.DataFrame({"ep": ep, "enmo": enmo.to_numpy(),
                        "gx": raw.x, "gy": raw.y, "gz": raw.z}).groupby("ep").agg(
        enmo=("enmo", "mean"), gx=("gx", "mean"), gy=("gy", "mean"),
        gz=("gz", "mean"), n=("enmo", "size"))

    grid = np.arange(ep.max() + 1)
    agg = agg.reindex(grid)
    agg["time"] = t0 + pd.to_timedelta(grid * cfg.epoch_sec, unit="s")
    agg["coverage"] = (agg["n"].fillna(0) / expected).clip(0, 1)

    gv = agg[["gx", "gy", "gz"]].to_numpy()
    norm = np.linalg.norm(gv, axis=1, keepdims=True)
    unit = np.divide(gv, norm, out=np.full_like(gv, np.nan), where=norm > 0)
    dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1, 1)
    agg["tilt_change"] = np.concatenate([[0.0], np.degrees(np.arccos(dot))])
    return agg.reset_index(drop=True)


# --- sleep-period detection -------------------------------------------------
def _detect_window(ep, cfg):
    """Return (start, end) epoch indices of the sleep window, or (None, None).

    The window is the longest sustained-stillness block. Brief motion (an
    awakening) or missing data shorter than bridge_min stays inside the window
    rather than ending it -- otherwise every awakening would truncate the night.
    """
    enmo = ep["enmo"].to_numpy()
    still = (enmo < cfg.still_enmo_mg) & (ep["tilt_change"].to_numpy() < cfg.tilt_thresh_deg)
    still = np.where(np.isnan(enmo), True, still)  # missing data doesn't break stillness

    w = max(round(cfg.block_min * 60 / cfg.epoch_sec), 1)
    frac = pd.Series(still.astype(float)).rolling(w, center=True, min_periods=1).mean()
    runs = list(_runs((frac >= 0.5).to_numpy()))
    if not runs:
        return None, None

    bridge = round(cfg.bridge_min * 60 / cfg.epoch_sec)
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= bridge:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return tuple(max(merged, key=lambda r: r[1] - r[0]))


# --- sleep/wake scoring -----------------------------------------------------
def _score(ep, s, e, cfg):
    """Label each epoch in [s, e] asleep (True) / awake (False). Missing epochs
    are treated as still (data_coverage separately flags them)."""
    act = ep["enmo"].iloc[s:e + 1].fillna(0.0).to_numpy()
    asleep = act < cfg.still_enmo_mg

    w = max(cfg.smooth_epochs | 1, 1)  # odd window for a symmetric majority vote
    smoothed = pd.Series(asleep.astype(float)).rolling(w, center=True, min_periods=1).mean()
    return (smoothed >= 0.5).to_numpy()


# --- metrics ----------------------------------------------------------------
def _empty_row(student_id, date, cfg):
    row = {c: None for c in OUTPUT_COLUMNS}
    row.update(student_id=student_id, date=date, data_coverage=0.0,
               quality_ok=False, algo_version=ALGO_VERSION,
               params_hash=cfg.params_hash())
    return row


def _metrics(ep, s, e, asleep, cfg, student_id, date):
    seg = ep.iloc[s:e + 1].reset_index(drop=True)
    per_epoch_min = cfg.epoch_sec / 60.0
    window_len = len(seg) * per_epoch_min
    total_sleep = float(asleep.sum() * per_epoch_min)
    longest_bout = max((b - a + 1 for a, b in _runs(asleep)), default=0) * per_epoch_min

    if asleep.any():
        onset = int(np.argmax(asleep))
        offset = int(np.max(np.where(asleep)[0]))
        awake_after = ~asleep[onset:]
        waso = float(awake_after.sum() * per_epoch_min)
        min_burst = max(round(cfg.awake_burst_min / per_epoch_min), 1)
        awakenings = sum((b - a + 1) >= min_burst for a, b in _runs(awake_after))
        onset_t, offset_t = seg["time"].iloc[onset], seg["time"].iloc[offset]
        midpoint = (onset_t + (offset_t - onset_t) / 2).strftime("%H:%M")
    else:
        waso, awakenings, onset_t, offset_t, midpoint = 0.0, 0, None, None, None

    coverage = float(seg["coverage"].mean())
    return {
        "student_id": student_id,
        "date": date,
        "sleep_onset": onset_t.strftime("%H:%M") if onset_t is not None else None,
        "sleep_offset": offset_t.strftime("%H:%M") if offset_t is not None else None,
        "window_len_min": round(window_len, 1),
        "total_sleep_min": round(total_sleep, 1),
        "longest_still_bout_min": round(longest_bout, 1),
        "num_awakenings_est": int(awakenings),
        "waso_total_min": round(waso, 1),
        "sleep_efficiency": round(100.0 * total_sleep / window_len, 1) if window_len else 0.0,
        "sleep_midpoint": midpoint,
        "data_coverage": round(coverage, 2),
        "quality_ok": bool(coverage >= cfg.min_coverage and window_len >= cfg.min_window_min),
        "algo_version": ALGO_VERSION,
        "params_hash": cfg.params_hash(),
    }


# --- result -----------------------------------------------------------------
@dataclass
class NightResult:
    """One night's metrics. Indexes like the row: result["total_sleep_min"]."""

    row: dict

    def __getitem__(self, key):
        return self.row[key]

    def explain(self):
        r = self.row
        if r["sleep_onset"] is None:
            msg = f"{r['date']}: no clear sleep period detected -- quality LOW."
        else:
            cov = int(r["data_coverage"] * 100)
            quality = "OK" if r["quality_ok"] else f"LOW (coverage {cov}%)"
            h, m = divmod(int(r["total_sleep_min"]), 60)
            msg = (f"{r['date']}: slept {r['sleep_onset']}-{r['sleep_offset']}, "
                   f"{h}h{m:02d}m asleep, {r['num_awakenings_est']} awakening(s), "
                   f"efficiency {r['sleep_efficiency']}%, coverage {cov}% -- "
                   f"quality {quality}.")
        print(msg)
        return msg


# --- public API -------------------------------------------------------------
def _split_nights(raw):
    """Group samples into local nights labelled by the EVENING date: the noon-to-
    noon window is shifted back 12h, so 9pm Aug 18 - 7am Aug 19 all belong to Aug 18."""
    return raw.groupby((raw["time"] - pd.Timedelta(hours=12)).dt.normalize())


def _analyze_group(raw_night, cfg, student_id, date):
    ep = _epoch_features(raw_night, cfg)
    s, e = _detect_window(ep, cfg)
    if s is None:
        return NightResult(_empty_row(student_id, date, cfg))
    asleep = _score(ep, s, e, cfg)
    return NightResult(_metrics(ep, s, e, asleep, cfg, student_id, date))


def analyze(source, config=None, drop_low_quality=False, column_map=None):
    """Analyze a phone-accelerometer CSV (path or DataFrame) of one or more nights.

    Returns one row per detected night as a DataFrame keyed on (student_id, date),
    where date is the night's evening date. The student id is inferred from the
    data, so none need be passed. See Config for advanced options.
    """
    cfg = config or Config()
    raw, sid = _load(source, column_map=column_map)
    rows = [_analyze_group(g, cfg, sid, k.date().isoformat()).row
            for k, g in _split_nights(raw) if len(g) > 1]
    df = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    if drop_low_quality and len(df):
        kept = df[df["quality_ok"]].reset_index(drop=True)
        kept.attrs["dropped_low_quality"] = len(df) - len(kept)
        return kept
    return df


def analyze_night(source, date=None, config=None, column_map=None):
    """Analyze one night and return a NightResult with .explain().
    With no date, the highest-coverage night is chosen."""
    cfg = config or Config()
    raw, sid = _load(source, column_map=column_map)
    results = [_analyze_group(g, cfg, sid, k.date().isoformat())
               for k, g in _split_nights(raw) if len(g) > 1]
    if not results:
        raise ValueError("No analyzable nights in the file.")
    if date is None:
        return max(results, key=lambda r: r.row.get("data_coverage") or 0)
    for r in results:
        if r.row["date"] == str(date):
            return r
    raise ValueError(f"Date {date} not found. Available: {[r.row['date'] for r in results]}")


def main(argv=None):
    """CLI: `sleepmetric analyze data.csv -o metrics.csv`
            `sleepmetric beiwe accel_folder/ -o metrics.csv --save-clean clean.csv`."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="sleepmetric", description="Phone accelerometer -> nightly sleep metrics.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze a tidy CSV of one or more nights")
    a.add_argument("csv")
    a.add_argument("-o", "--out", default="sleep_metrics.csv")
    a.add_argument("--drop-low-quality", action="store_true")

    b = sub.add_parser("beiwe", help="preprocess + analyze a Beiwe accelerometer export")
    b.add_argument("source", help="Beiwe accelerometer folder, glob, or CSV")
    b.add_argument("-o", "--out", default="sleep_metrics.csv")
    b.add_argument("--save-clean", default=None, help="also save the preprocessed CSV")

    args = parser.parse_args(argv)
    if args.cmd == "beiwe":
        df = analyze_beiwe(args.source, save_preprocessed=args.save_clean)
    else:
        df = analyze(args.csv, drop_low_quality=args.drop_low_quality)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} night(s) -> {args.out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
