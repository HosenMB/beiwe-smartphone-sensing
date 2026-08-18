# Smartphone Sensing: Human Activity & Sleep Analysis

A project for collecting passive smartphone sensor data with the
[Beiwe](https://www.beiwe.org/) research platform and turning it into
interpretable **sleep** and **human-activity** measures. Students install the
Beiwe app, contribute accelerometer data, and analyze it with the two Python
packages in this repository.

## Repository layout

```
.
├── README.md                     # this overview
├── Beiwe_Student_Setup_Guide.md  # install the app & register (participants)
├── Beiwe_Data/                   # raw Beiwe exports
├── Output/                       # analysis results are written here
├── run_sleep.py                  # sleep analysis
├── run_har.py                    # activity analysis
├── Sleep_Metrics/                # nightly sleep metrics    → Sleep_Metrics/README.md
└── Human_Activity/               # activity recognition     → Human_Activity/README.md
```

## Prerequisites (macOS & Windows)

1. **Install Python 3.9 or newer.**
   - Windows: download from [python.org](https://www.python.org/downloads/) and
     tick *"Add Python to PATH"* during setup.
   - macOS: download from python.org, or `brew install python`.
2. **Open a terminal in this folder** (the one containing this README).
   - Windows: **PowerShell** (Shift-right-click the folder → *Open in Terminal*).
   - macOS: **Terminal** (right-click the folder in Finder → *New Terminal at Folder*).
3. In the commands below, use **`python`** on Windows. On macOS, , use **`python3`**.

## Getting started

1. **Collect data.** Follow the [student setup guide](Beiwe_Student_Setup_Guide.md)
   to install the Beiwe app and register. The app records accelerometer data in
   the background.
2. **Export data** from the Beiwe dashboard into `Beiwe_Data/` — one folder per
   participant, each containing an `accelerometer/` folder of CSVs.
3. **Install a tool and run it** (from this folder):

   ```bash
   # Sleep
   python -m pip install -e Sleep_Metrics
   python run_sleep.py

   # Human activity
   python -m pip install -e Human_Activity
   python run_har.py
   ```

   Results are written to `Output/`. See each package's README for full details
   ([Sleep_Metrics](Sleep_Metrics/README.md), [Human_Activity](Human_Activity/README.md)).

## Data format

Both tools read the Beiwe **accelerometer** stream. An export contains one folder
per participant, each holding hourly CSV files named
`YYYY-MM-DD HH_00_00+00_00.csv` (UTC):

| Column | Meaning |
|--------|---------|
| `timestamp` | Unix time in milliseconds (UTC) |
| `UTC time` | ISO-8601 timestamp (UTC) |
| `accuracy` | sensor accuracy flag |
| `x`, `y`, `z` | acceleration per axis (g on iOS, m/s² on Android) |

```
Beiwe_Data/
└── <participant_id>/
    └── accelerometer/
        ├── 2024-09-24 16_00_00+00_00.csv
        └── ...
```

## Requirements

Python 3.9+. All Python dependencies (`numpy`, `pandas`, `scipy`, and, for
Human_Activity, `scikit-learn`, `pytz`, `tqdm`, `geopy`) install automatically
with each package via the commands above.

## License

MIT.
