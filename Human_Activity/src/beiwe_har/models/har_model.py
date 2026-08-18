"""Predict activity labels from ECDF features using the packaged RandomForest."""
import os
import joblib
import pandas as pd
from importlib.resources import files

# MotionSense classes
_ID2LABEL = {0: "dws", 1: "ups", 2: "wlk", 3: "jog", 4: "std", 5: "sit"}
_META = ["user", "window_start_time", "session_id"]


def predict_har_from_ecdf(features_file: str) -> pd.DataFrame:
    """Predict per-window activity from an ECDF-features CSV -> `*_predictions.csv`."""
    with files("beiwe_har.models").joinpath("random_forest_ecdf.pkl").open("rb") as f:
        model = joblib.load(f)

    df = pd.read_csv(features_file)
    keep = [c for c in ("user", "window_start_time") if c in df.columns]
    X = df.drop(columns=[c for c in _META if c in df.columns])

    out = df[keep].copy()
    out["predicted_label"] = [_ID2LABEL.get(int(y), str(y)) for y in model.predict(X)]

    output_file = os.path.splitext(features_file)[0].replace("_ECDF_features", "") + "_predictions.csv"
    out.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")
    return out
