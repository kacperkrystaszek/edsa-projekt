import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from import_data import import_from_files, get_filenames, get_tablename
from read_data import read_from_db

def denoise(df: pd.DataFrame, derivative_filter: bool = True) -> tuple[pd.DataFrame, float]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    dt: float = df.index.to_series().diff().median()
    fs = 1/dt
    lowcut = 5
    highcut = 15
    order = 3

    b, a = signal.butter(N=order, Wn=[lowcut, highcut], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, df['value'].values)
    if derivative_filter:
        h = np.array([1, 2, 0, -2, -1])
        filtered = signal.filtfilt(h, 1, filtered)
    df['filtered'] = filtered

    return df, fs

def fourier(values, fs) -> tuple:
    fft_values = np.fft.fft(values)
    fft_frequencies = np.fft.fftfreq(len(values), d=1/fs)

    mask = fft_frequencies > 0
    fft_frequencies = fft_frequencies[mask]
    fft_values = np.abs(fft_values[mask])
    
    return fft_frequencies, fft_values

def plot_fourier(fft_frequencies, fft_values, title, filename) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(fft_frequencies, fft_values)
    plt.xlabel('Freq[Hz]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.savefig(f"results/{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

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
    
def plot_results(df: pd.DataFrame, r_peaks: list, p_peaks: list, t_peaks: list, tablename: str) -> None:
    t = df.index.to_numpy()
    min_secs = 200
    max_secs = 210
    mask = (t >= min_secs) & (t <= max_secs)

    values = df['value'].to_numpy()
    filtered = df['filtered'].to_numpy()
    
    plt.figure(figsize=(12,4))
    plt.plot(t[mask], values[mask], label="Surowy sygnal")
    plt.plot(t[mask], filtered[mask], label="Wyfiltrowany sygnal")

    rpq_mask = lambda x: [i for i in x if t[i] >= min_secs and t[i] <= max_secs]
    
    r_window = rpq_mask(r_peaks)
    p_window = rpq_mask(p_peaks)
    t_window = rpq_mask(t_peaks)
    
    plt.scatter(t[r_window], filtered[r_window], color="red", label="R")
    plt.scatter(t[p_window], filtered[p_window], color="green", label="P")
    plt.scatter(t[t_window], filtered[t_window], color="blue", label="T")

    plt.xlabel("Time[s]")
    plt.ylabel('EKG')
    plt.title(f"R, P, T peaks in {min_secs} - {max_secs} seconds - {tablename}")
    plt.legend()
    plt.savefig(f"results/ekg_{tablename}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
def _main(filename: str, plot_fourier: bool = True) -> None:
    tablename = get_tablename(filename)
    df = read_from_db(tablename=tablename)
    if df is None:
        print("Error")
        return
    df: pd.DataFrame
    df, fs = denoise(df)

    
    if plot_fourier:
        fft_freq, fft_values = fourier(df['value'].values, fs)
        plot_fourier(fft_freq, fft_values, f"FFT before denoise - {tablename}", f"fft_no_denoised_{tablename}")
        fft_freq, fft_values = fourier(df['filtered'].values, fs)
        plot_fourier(fft_freq, fft_values, f"FFT after denoise - {tablename}", f"fft_denoised_{tablename}")

    filtered_data = df['filtered'].to_numpy()
    min_dist = int(200 * 60 / 180)
    
    height = np.mean(filtered_data) + np.std(filtered_data)
    r_peaks, _ = signal.find_peaks(filtered_data, distance=min_dist, height=height)

    p_peaks, t_peaks = find_p_and_t(fs, filtered_data, r_peaks)

    r_to_r_times = np.diff(pd.to_datetime(df.index.to_numpy()[r_peaks]).astype("int64"))
    bpm_avg = 60 / np.mean(r_to_r_times)

    print(f"Avg BPM: {bpm_avg}")

    plot_results(df, r_peaks, p_peaks, t_peaks, tablename)
    
def main() -> None:
    # import_from_files()
    
    filenames = get_filenames()
    for filename in filenames:
        print(filename)
        _main(filename, False)

if __name__ == "__main__":
    main()