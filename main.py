import numpy as np
import pandas as pd
from scipy import signal
from import_data import get_filenames, get_tablename, import_from_files
from plots import plot_fourier, plot_results
from signal_processing import normalize, denoise, fourier, find_p_and_t, find_r_peaks, calculate_bpm
from read_data import read_from_db

def _main(filename: str, plot_f: bool = True) -> None:
    tablename = get_tablename(filename)
    df = read_from_db(tablename=tablename)
    if df is None:
        print("Error")
        return
    df: pd.DataFrame
    df = normalize(df)
    df, fs = denoise(df)
    
    if plot_f:
        fft_freq, fft_values = fourier(df['value'].values, fs)
        plot_fourier(fft_freq, fft_values, f"FFT before denoise - {tablename}", f"fft_no_denoised_{tablename}")
        fft_freq, fft_values = fourier(df['filtered'].values, fs)
        plot_fourier(fft_freq, fft_values, f"FFT after denoise - {tablename}", f"fft_denoised_{tablename}")

    filtered_data = df['filtered'].to_numpy()

    r_peaks = find_r_peaks(fs, filtered_data, tablename)
    r_times = df.index.to_numpy()[r_peaks].astype("float")
        
    p_peaks, t_peaks = find_p_and_t(fs, filtered_data, r_peaks)

    bpm_avg = calculate_bpm(r_times)

    print(f"Avg BPM: {bpm_avg:.1f}")

    plot_results(df, r_peaks, p_peaks, t_peaks, tablename)
    
def main() -> None:
    # import_from_files()
    
    filenames = get_filenames()
    for filename in filenames:
        print(filename)
        _main(filename, False)

if __name__ == "__main__":
    main()