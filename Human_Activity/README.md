# beiwe_har — Human Activity Recognition from Beiwe accelerometer

An end-to-end pipeline: a raw [Beiwe](../README.md) study export → per-window
activity predictions and activity summaries. Part of the Beiwe smartphone-sensing
project.

## Install

From the project's top folder (see the
[prerequisites](../README.md#prerequisites-macos--windows) for macOS/Windows notes):

```bash
python -m pip install -e Human_Activity
```

## Usage

The simplest way is the ready-made script, which analyzes every participant in
`Beiwe_Data/` and writes results to `Output/`:

```bash
python run_har.py
```

Equivalently, from your own code or the command line:

```python
import beiwe_har as bh
bh.run_pipeline("Beiwe_Data", output_dir="Output")   # add uci_har=True for UCI-HAR features
```

```bash
beiwe-har Beiwe_Data -o Output --uci-har
```

## What it does (per participant)

1. Combine the raw `<participant>/accelerometer/*.csv` files and convert
   UTC → US/Eastern.
2. Split into sessions on gaps > 10 s, **resample each session to 50 Hz**, and cut
   2-second windows (50 % overlap). Gravity is removed (low-pass) so the signal
   matches the model's training data.
3. **ECDF features** (78 per window) → **Random Forest** → one activity label per
   window. Classes (MotionSense): `dws` = downstairs, `ups` = upstairs,
   `wlk` = walk, `jog`, `std` = stand, `sit`.
4. **Per-minute** and **day-wise** activity summaries (`Low` / `Medium` / `High`).

Optional: UCI-HAR features (`uci_har=True` / `--uci-har`); and a daily
activity-level table (needs gyroscope + GPS streams, per morning/day/night epoch).

## Outputs (written to `Output/`)

| File | Contents |
|------|----------|
| `acc_dataset.csv` | combined per-sample accelerometer (user, timestamp, x, y, z, magnitude) |
| `acc_dataset_segmented.csv` | 2-second windows at 50 Hz |
| `acc_dataset_ECDF_features.csv` | 78 ECDF features per window |
| `acc_dataset_predictions.csv` | predicted activity per window |
| `acc_aggregated.csv` | per-minute activity level |
| `daily_summary.csv` | day-wise summary |

## Limitations

The model was trained on the MotionSense dataset at a true 50 Hz. Beiwe
accelerometer is sampled more slowly and irregularly, so each session is
**upsampled** to 50 Hz; the interpolated high-frequency detail is not real, so
predictions should be treated as approximate. See the
[data format](../README.md#data-format).

## License

MIT. Author: Md Biplob Hosen (mhosen1@umbc.edu).
