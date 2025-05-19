import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline
from Preprocessing import segment_signal

"=== EMD ==="
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

"=== EEMD ==="
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

"=== CEEMDAN ==="
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