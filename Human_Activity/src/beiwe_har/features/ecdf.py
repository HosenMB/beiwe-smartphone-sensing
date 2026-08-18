"""ECDF feature extraction (78 features = 25 components + mean, per axis).

Matches the representation the packaged RandomForest was trained on, so the
`ecdfRep` math must not change.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

_META = ["user", "window_start_time", "session_id"]


def extract_ECDF_features(input_file, frequency=50, duration_in_seconds=2, num_components=25):
    """Extract ECDF features from a segmented accelerometer CSV -> `*_ECDF_features.csv`."""
    window_size = int(frequency * duration_in_seconds)

    def ecdfRep(data, components):
        m = np.mean(data, axis=0)
        data = np.sort(data, axis=0)
        data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
        return np.hstack((data.flatten(), m))

    df = pd.read_csv(input_file)
    meta = [c for c in _META if c in df.columns]
    meta_cols = df[meta].copy()
    X = df.drop(columns=meta).values.reshape(-1, window_size, 3)

    features = np.zeros((X.shape[0], (num_components + 1) * 3))
    for i in tqdm(range(X.shape[0]), desc=os.path.basename(input_file)):
        features[i] = ecdfRep(X[i], num_components)

    df_out = pd.concat([meta_cols.reset_index(drop=True), pd.DataFrame(features)], axis=1)
    base = os.path.splitext(os.path.basename(input_file))[0].replace("_segmented", "")
    output_file = os.path.join(os.path.dirname(input_file), f"{base}_ECDF_features.csv")
    df_out.to_csv(output_file, index=False)
    print(f"Saved ECDF features to: {output_file}")
    return df_out
