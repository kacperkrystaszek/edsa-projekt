import numpy as np
import pandas as pd
from scipy import signal


def normalize(df: pd.DataFrame):
    # Normalize signal to [-1; 1]
    df['value_copy'] = df['value']
    df['value'] = df['value'] - df['value'].mean()
    df['value'] = df['value'] / np.max(np.abs(df['value']))
    return df

def denoise(df: pd.DataFrame, derivative_filter: bool = True) -> tuple[pd.DataFrame, float]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    dt = np.diff(df.index.to_numpy()).astype("float").mean()
    fs = 1/dt # 200
    # pan tompkins algorithm
    lowcut = 5 
    highcut = 15
    n = 3
    Wn = [lowcut, highcut]

    b, a = signal.butter(N=n, Wn=Wn, btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, df['value'].values)
    
    if derivative_filter:
        # pan tompkins algorithm
        h = np.array([1, 2, 0, -2, -1])
        filtered = signal.filtfilt(h, 1, filtered)

    # normalization
    filtered = filtered / np.max(np.abs(filtered))
    df['filtered'] = filtered

    return df, fs

def fourier(values, fs) -> tuple:
    fft_values = np.fft.fft(values)
    fft_frequencies = np.fft.fftfreq(len(values), d=1/fs)

    mask = fft_frequencies > 0
    fft_frequencies = fft_frequencies[mask]
    fft_values = np.abs(fft_values[mask])
    
    return fft_frequencies, fft_values

def find_r_peaks(fs, filtered_data, tablename):
    # peak finding parameters
    min_dist = round(200 * fs / 1000)
    height = np.mean(filtered_data) + 0.7 * np.std(filtered_data)
    PROMINENCES = {
        "no_move": 0.7,
        "in_move": 0.3
    }
    prominence_key = next(filter(lambda key: key in tablename, PROMINENCES.keys()), "")
    prominence = PROMINENCES.get(prominence_key)
    
    r_peaks, _ = signal.find_peaks(filtered_data, distance=min_dist, height=height, prominence=prominence)
    return r_peaks

def find_p_and_t(fs, filtered_data, r_peaks):
    p_peaks = []
    t_peaks = []

    for r in r_peaks:
        start_p = max(0, r - int(0.12*fs))
        end_p = max(0, r - int(0.05*fs))
        p_window = filtered_data[start_p: end_p]
        if len(p_window):
            p = np.argmax(p_window) + start_p
            p_peaks.append(p)

        start_t = r + int(0.08 * fs)
        end_t = min(len(filtered_data), r + int(0.24 * fs))
        t_window = filtered_data[start_t: end_t]
        if len(t_window):
            t = np.argmax(t_window) + start_t
            t_peaks.append(t)

    return p_peaks, t_peaks

def calculate_bpm(peak_times):
    r_to_r_intervals = np.diff(peak_times)
    bpm = 60 / np.mean(r_to_r_intervals)
    return bpm
