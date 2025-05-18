import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline


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



"=== Signal Decomposition ==="
def compute_envelope_mean(signal, extrema_indices):
    if len(extrema_indices) < 2:
        return np.zeros_like(signal)
    x = extrema_indices
    y = signal[extrema_indices]
    cs = CubicSpline(x, y, extrapolate=True)
    return cs(np.arange(len(signal)))

def calculate_sd(h_prev, h_curr):
    eps = 1e-10  # avoid division by zero
    return np.sum(((h_prev - h_curr)**2) / (h_prev**2 + eps)) / len(h_prev)

def is_imf(signal):
    zero_crossings = np.sum(signal[:-1] * signal[1:] < 0)
    maxima = argrelextrema(signal, np.greater)[0]
    minima = argrelextrema(signal, np.less)[0]
    num_extrema = len(maxima) + len(minima)
    return abs(zero_crossings - num_extrema) <= 1

def emd(signal, max_imfs=10, max_siftings=100, sd_thresh=0.3, visualize=True):
    imfs = []
    residue = signal.copy()

    for imf_idx in range(max_imfs):
        h = residue.copy()
        for _ in range(max_siftings):
            max_peaks = argrelextrema(h, np.greater)[0]
            min_peaks = argrelextrema(h, np.less)[0]
            
            if len(max_peaks) < 2 or len(min_peaks) < 2:
                break

            upper_env = compute_envelope_mean(h, max_peaks)
            lower_env = compute_envelope_mean(h, min_peaks)
            mean_env = (upper_env + lower_env) / 2

            h_new = h - mean_env
            sd = calculate_sd(h, h_new)

            if is_imf(h_new) and sd < sd_thresh:
                break
            h = h_new

        imfs.append(h_new)
        residue = residue - h_new

        if np.std(residue) < 1e-10:
            break

    imfs.append(residue)  
    return np.array(imfs)

def eemd(signal, emd_func, noise_strength=0.2, num_ensembles=50, max_imfs=10):
    N = len(signal)
    I = num_ensembles
    eps = noise_strength

    ensemble_imfs = []

    for i in range(I):
        noise = np.random.normal(0, 1, N)
        x_i = signal + eps * noise
        imfs = emd_func(x_i, max_imfs=max_imfs)
        ensemble_imfs.append(imfs)

    
    num_actual_imfs = max([imf.shape[0] for imf in ensemble_imfs])
    result = []

    for k in range(num_actual_imfs):
        imfs_k = []
        for imf in ensemble_imfs:
            if k < imf.shape[0]:
                imfs_k.append(imf[k])
            else:
                imfs_k.append(np.zeros(N))
        result.append(np.mean(imfs_k, axis=0))

    return np.array(result)


def ceemdan(signal, emd_func, noise_strength=0.2, num_ensembles=50, max_imfs=10):
    N = len(signal)
    I = num_ensembles
    eps = noise_strength

    r = signal.copy()
    imfs_result = []

    # IMF1
    imf1_list = []
    for i in range(I):
        noise = np.random.normal(0, 1, N)
        x_i = signal + eps * noise
        imfs = emd_func(x_i, max_imfs=1)
        imf1_list.append(imfs[0])
    IMF1 = np.mean(imf1_list, axis=0)
    imfs_result.append(IMF1)
    r = signal - IMF1

    # IMF2 onwards
    for k in range(1, max_imfs):
        imf_k_list = []
        for i in range(I):
            noise = np.random.normal(0, 1, N)
            r_i = r + eps * emd_func(noise, max_imfs=1)[0]
            imfs = emd_func(r_i, max_imfs=1)
            imf_k_list.append(imfs[0])
        IMF_k = np.mean(imf_k_list, axis=0)
        imfs_result.append(IMF_k)
        r = r - IMF_k

        if np.std(r) < 1e-10:
            break

    imfs_result.append(r)
    return np.array(imfs_result)


def select_high_corr_imfs(signal, imfs, top_n=3):
    corrs = [np.corrcoef(signal, imf)[0, 1] for imf in imfs]
    sorted_indices = np.argsort(np.abs(corrs))[::-1]  
    return imfs[sorted_indices[:top_n]]


def segment_imfs(imfs, frame_size=1024):
    segmented = []
    for imf in imfs:
        segments = segment_signal(imf, frame_size=frame_size, overlap=0)  
        segmented.extend(segments)
    return segmented




"=== Feature extraction ==="
def extract_20_features(signal, sampling_rate=12000):
    N = len(signal)

    # Time domain features
    xmax = np.max(signal)                        # 1. Maximum
    xmin = np.min(signal)                        # 2. Minimum
    xmean = np.mean(signal)                      # 3. Mean
    xstd = np.std(signal, ddof=1)                # 4. Standard Deviation
    xrms = np.sqrt(np.mean(signal**2))           # 5. Root Mean Square
    xskew = np.sum((signal - xmean)**3) / ((N - 1) * xstd**3)  # 6. Skewness
    xkurt = np.sum((signal - xmean)**4) / ((N - 1) * xstd**4)  # 7. Kurtosis
    xshape = xrms / np.mean(np.abs(signal))      # 8. Shape Factor

    # Frequency domain
    freqs = np.fft.rfftfreq(N, d=1/sampling_rate)
    fft_vals = np.fft.rfft(signal)
    ps = np.abs(fft_vals) ** 2  # Power Spectrum
    ps_norm = ps / np.sum(ps) if np.sum(ps) != 0 else ps

    mean_freq = np.sum(freqs * ps_norm)                      # 9. Mean Frequency
    power_bw = np.sqrt(np.sum((freqs - mean_freq)**2 * ps_norm))  # 10. Power Bandwidth

    crest = xmax / xrms if xrms != 0 else 0                  # 11. Crest Factor
    form = xrms / np.mean(np.abs(signal))                    # 12. Form Factor
    peak_to_peak = xmax - xmin                               # 13. Peak-to-Peak
    xvar = np.var(signal)                                    # 14. Variance
    xmed = np.median(signal)                                 # 15. Median
    mean_square = xrms**2                                    # 16. Mean Square
    margin = xmax / np.sqrt(mean_square) if mean_square != 0 else 0  # 17. Margin
    impulse = xrms**2 / np.mean(np.abs(signal))              # 18. Impulse

    # 19. Median Frequency (frequency where cumulative power reaches 50%)
    cumulative_power = np.cumsum(ps_norm)
    median_freq = freqs[np.searchsorted(cumulative_power, 0.5)]

    freq_center = np.sum(freqs * ps_norm)                    # 20. Frequency Center (same as mean frequency)

    return {
        'Maximum': xmax,
        'Minimum': xmin,
        'Mean': xmean,
        'Standard Deviation': xstd,
        'Root Mean Square': xrms,
        'Skewness': xskew,
        'Kurtosis': xkurt,
        'Shape Factor': xshape,
        'Mean Frequency': mean_freq,
        'Power Bandwidth': power_bw,
        'Crest Factor': crest,
        'Form Factor': form,
        'Peak-to-Peak': peak_to_peak,
        'Variance': xvar,
        'Median': xmed,
        'Mean Square': mean_square,
        'Margin': margin,
        'Impulse': impulse,
        'Median Frequency': median_freq,
        'Frequency Center': freq_center
    }




"=== Run ==="
folder_path = r"E:\ON CLASS\Y4_S2\Thesis\Data\CWRU"
signals = load_de_signals(folder_path)

train_set = {}
test_set = {}

for filename, signal in signals.items():
    signal_cleaned = remove_outliers(signal)
    signals_normalized = normalize(signal_cleaned)

    X_train, X_test = train_test_split(signals_normalized, test_size=0.3, random_state=42)
    train_set[filename] = X_train
    test_set[filename] = X_test


for filename, train_signal in train_set.items():
    print(f"Processing file: {filename}")

    segments = segment_signal(train_signal, frame_size=1024, overlap=0)

    for i, segment in enumerate(segments):  
        # imfs = emd(segment)
        #imfs = eemd(segment, emd_func=emd)
        imfs = ceemdan(segment, emd_func=emd)


        plt.figure(figsize=(12, 2 * (len(imfs) + 1)))
        plt.subplot(len(imfs) + 1, 1, 1)
        plt.plot(segment)
        plt.title(f"{filename} - Segment {i} - Original Signal")

        for j, imf in enumerate(imfs):
            plt.subplot(len(imfs) + 1, 1, j + 2)
            plt.plot(imf)
            plt.title(f"IMF {j + 1}")

        plt.tight_layout()
        plt.show()
