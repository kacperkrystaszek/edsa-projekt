from matplotlib import pyplot as plt
import pandas as pd


def plot_results(df: pd.DataFrame, r_peaks: list, p_peaks: list, t_peaks: list, tablename: str) -> None:
    t = df.index.to_numpy()
    min_secs = 200
    max_secs = 210
    mask = (t >= min_secs) & (t <= max_secs)

    values = df['value'].to_numpy()
    filtered = df['filtered'].to_numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t[mask], values[mask], label="Surowy sygnal")
    axes[0].set_title("Raw EKG signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time[s]")

    axes[1].plot(t[mask], filtered[mask], color="orange", label="Processed EKG signal")

    rpq_mask = lambda x: [i for i in x if t[i] >= min_secs and t[i] <= max_secs]
    
    r_window = rpq_mask(r_peaks)
    p_window = rpq_mask(p_peaks)
    t_window = rpq_mask(t_peaks)
    
    axes[1].scatter(t[r_window], filtered[r_window], color="red", label="R")
    axes[1].scatter(t[p_window], filtered[p_window], color="green", label="P")
    axes[1].scatter(t[t_window], filtered[t_window], color="blue", label="T")

    axes[1].set_xlabel("Time[s]")
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f"Processed EKG signal with R, P, T peaks in <{min_secs};{max_secs}> seconds - {tablename}")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"results/ekg_{tablename}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_fourier(fft_frequencies, fft_values, title, filename) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(fft_frequencies, fft_values)
    plt.xlabel('Freq[Hz]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.savefig(f"results/{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()
