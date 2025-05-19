import numpy as np



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

















