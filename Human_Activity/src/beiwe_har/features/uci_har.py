# UCI-HAR feature extraction 

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import entropy, iqr, skew, kurtosis
from numpy.fft import fft
from numpy.linalg import lstsq
from scipy.signal import butter, filtfilt


def extract_UCI_HAR_features(input_file, frequency=50, duration_in_seconds=2):
    """
    Extract UCI HAR features from segmented CSV file.

    Parameters:
    - input_file: str, path to a segmented CSV file
    - frequency: int, sampling frequency in Hz (default: 50)
    - duration_in_seconds: int or float, window size in seconds (default: 2)
    """
    window_size = int(frequency * duration_in_seconds)

    def low_pass_filter(signal, cutoff=0.3, fs=10, order=1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def signal_entropy(signal):
        hist, _ = np.histogram(signal, bins=10)
        return entropy(hist + 1)

    def autoregression_coeffs(x, order=4):
        N = len(x)
        if N <= order:
            return [0] * order
        X = np.column_stack([x[i:N - order + i] for i in range(order)])
        y = x[order:]
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        return coeffs[:order]

    def correlation(a, b):
        return np.corrcoef(a, b)[0, 1]

    def vector_angle(a, b):
        a, b = np.array(a), np.array(b)
        dot_product = np.dot(a, b)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

    def compute_band_energy(signal_fft, bands):
        band_energy = []
        for start, end in bands:
            energy = np.sum(signal_fft[start:end] ** 2)
            band_energy.append(energy)
        return band_energy

    def extract_features_from_window(window, sampling_rate):
        features = {}
        n = len(window)
        accel_axes = ['X', 'Y', 'Z']

        gravity = {axis: low_pass_filter(window[axis].values, fs=sampling_rate) for axis in accel_axes}
        body = {axis: window[axis].values - gravity[axis] for axis in accel_axes}
        jerk = {axis: np.gradient(body[axis], 1 / sampling_rate) for axis in accel_axes}

        def extract_for_signal(signal, label):
            for axis in accel_axes:
                values = signal[axis]
                features[f'{label}_{axis}_mean'] = np.mean(values)
                features[f'{label}_{axis}_std'] = np.std(values)
                features[f'{label}_{axis}_mad'] = np.mean(np.abs(values - np.mean(values)))
                features[f'{label}_{axis}_min'] = np.min(values)
                features[f'{label}_{axis}_max'] = np.max(values)
                features[f'{label}_{axis}_energy'] = np.sum(values ** 2) / n
                features[f'{label}_{axis}_iqr'] = iqr(values)
                features[f'{label}_{axis}_entropy'] = signal_entropy(values)
                ar_coeffs = autoregression_coeffs(values)
                for i, coeff in enumerate(ar_coeffs):
                    features[f'{label}_{axis}_arCoeff_{i+1}'] = coeff

                fft_vals = np.abs(fft(values))
                freq_bins = len(fft_vals)
                features[f'{label}_{axis}_meanFreq'] = np.sum(np.arange(freq_bins) * fft_vals) / np.sum(fft_vals)
                features[f'{label}_{axis}_maxInds'] = np.argmax(fft_vals)
                features[f'{label}_{axis}_skew'] = skew(values)
                features[f'{label}_{axis}_kurtosis'] = kurtosis(values)

                bands = [(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]
                band_energies = compute_band_energy(fft_vals, bands)
                for i, be in enumerate(band_energies):
                    features[f'{label}_{axis}_bandEnergy_{i+1}'] = be

            features[f'{label}_sma'] = np.sum(np.abs(signal['X']) + np.abs(signal['Y']) + np.abs(signal['Z'])) / n
            features[f'{label}_corr_X_Y'] = correlation(signal['X'], signal['Y'])
            features[f'{label}_corr_X_Z'] = correlation(signal['X'], signal['Z'])
            features[f'{label}_corr_Y_Z'] = correlation(signal['Y'], signal['Z'])

        extract_for_signal(body, 'body')
        extract_for_signal(gravity, 'gravity')
        extract_for_signal(jerk, 'jerk')

        body_mean = [np.mean(body[axis]) for axis in accel_axes]
        gravity_mean = [np.mean(gravity[axis]) for axis in accel_axes]
        jerk_mean = [np.mean(jerk[axis]) for axis in accel_axes]

        features['angle_body_gravity'] = vector_angle(body_mean, gravity_mean)
        features['angle_jerk_gravity'] = vector_angle(jerk_mean, gravity_mean)

        return features

    # Load segmented data
    df = pd.read_csv(input_file)

    # Preserve metadata columns (keep session_id if present)
    meta_names = [c for c in ["user", "window_start_time", "session_id"] if c in df.columns]
    if not {"user", "window_start_time"}.issubset(meta_names):
        # fallback to original behavior if names are unexpected
        meta_cols = df.iloc[:, :2].copy()
        signal_data = df.iloc[:, 2:].copy()
    else:
        meta_cols = df[meta_names].copy()
        signal_data = df.drop(columns=meta_names).copy()

    # Reshape segmented signals to (num_windows, window_size, 3)
    try:
        X = signal_data.values.reshape(-1, window_size, 3)
    except Exception as e:
        raise ValueError(f"Error reshaping {input_file}: {e}")

    feature_rows = []
    for i in tqdm(range(X.shape[0]), desc=os.path.basename(input_file)):
        window = pd.DataFrame(X[i], columns=["X", "Y", "Z"])
        features = extract_features_from_window(window, sampling_rate=frequency)
        row = meta_cols.iloc[i].to_dict()
        row.update(features)
        feature_rows.append(row)

    df_out = pd.DataFrame(feature_rows)
    output_file = input_file.replace("_segmented.csv", "_UCI_HAR_features.csv")
    df_out.to_csv(output_file, index=False)
    print(f"Saved UCI HAR features to: {output_file}")
    return df_out
