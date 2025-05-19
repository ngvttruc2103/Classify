import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

"=== PreProcessing ==="
# Load DE_time and FE_time signals
def load_de_signals(folder_path):
    signals = {}
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            path = os.path.join(folder_path, file)
            data = scipy.io.loadmat(path)
            for key in data:
                if "DE_time" in key:
                    signal = data[key].squeeze()
                    signals[file] = signal
    return signals

# Segment signals
def segment_signal(signal, frame_size=2048, overlap=0.5):
    step = int(frame_size * (1 - overlap))
    return [signal[i:i+frame_size] for i in range(0, len(signal) - frame_size + 1, step)]

# Check for missing values
def check_missing_values(signal):
    return np.isnan(signal).sum(), np.isinf(signal).sum()

# Remove outliers using 3-sigma rule
def remove_outliers(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    mask = (signal > mean - 3*std) & (signal < mean + 3*std)
    return signal[mask] if np.sum(mask) >= 100 else signal

# Normalize signals
def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

# Plot signals before and after preprocessing
def plot_signals(original, cleaned, title):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(original, color='blue')
    axs[0].set_title(f"{title} - Original Signal")
    axs[1].plot(cleaned, color='green')
    axs[1].set_title(f"{title} - After Outlier Removal")
    plt.tight_layout()
    plt.show()