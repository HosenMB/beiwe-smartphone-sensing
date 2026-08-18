# Nightly sleep metrics from a phone accelerometer

Interpretable, rule-based nightly sleep metrics from smartphone accelerometer
data.

All timestamps are converted to **Maryland local time** (`America/New_York`,
daylight-saving aware). A *night* is the local noon-to-noon window labelled by
its **evening date**, so 9 pm Aug 18 – 7 am Aug 19 is reported as `2024-08-18`.

## Install

From the project's top folder (see the
[prerequisites](../README.md#prerequisites-macos--windows) for macOS/Windows notes):

```bash
python -m pip install -e Sleep_Metrics
```

## Usage

The simplest way is the ready-made script, which analyzes one participant and
writes results to `Output/`:

```bash
python run_sleep.py
```

To use it in your own code:

```python
import sleepmetric as sm

# One summary row per night for a single participant's accelerometer folder.
df = sm.analyze_beiwe("Beiwe_Data/gp83emoi/accelerometer",
                      save_preprocessed="Output/gp83emoi_clean.csv")
df.to_csv("Output/gp83emoi_sleep_metrics.csv", index=False)
print(df.to_string(index=False))

# Inspect one night in plain English.
sm.analyze_night("Output/gp83emoi_clean.csv", date="2024-09-20").explain()
```

Point at a **single** participant's `accelerometer/` folder (see the
[data format](../README.md#data-format)). A tidy CSV with `timestamp` and
`x, y, z` columns also works via `sm.analyze()`.

## Output

One row per night:

| Column | Meaning |
|--------|---------|
| `student_id`, `date` | participant and night (evening date) |
| `sleep_onset`, `sleep_offset` | sleep start / end (local `HH:MM`) |
| `total_sleep_min` | minutes asleep |
| `longest_still_bout_min` | longest unbroken stillness |
| `num_awakenings_est` | awakenings after onset |
| `waso_total_min` | minutes awake after first onset |
| `sleep_efficiency` | asleep ÷ time-in-window × 100 |
| `sleep_midpoint` | clock midpoint of sleep |
| `data_coverage` | fraction of the night actually recorded |
| `quality_ok` | passes coverage + minimum-duration checks |

Each row also carries `window_len_min`, `algo_version`, and `params_hash` for
reproducibility.

## Limitations

Quiet wakefulness (lying still but awake) is scored as sleep, so efficiency is an
over-estimate. Sleep stages (light / deep / REM) are not produced — they are not
derivable from accelerometer alone. Nights with low `data_coverage` are flagged
`quality_ok = False` and should be treated cautiously.

## License

MIT.
