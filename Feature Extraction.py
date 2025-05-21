# Feature_Extraction.py
import numpy as np
import pandas as pd
from Preprocessing import segment_signal
from EMD_EEMD_CEEMDAN import emd, eemd, ceemdan, select_high_corr_imfs

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

    mean_freq = np.sum(freqs * ps_norm)          # 9. Mean Frequency
    power_bw = np.sqrt(np.sum((freqs - mean_freq)**2 * ps_norm))  # 10. Power Bandwidth
    
    crest = xmax / xrms if xrms != 0 else 0      # 11. Crest Factor
    form = xrms / np.mean(np.abs(signal))        # 12. Form Factor
    peak_to_peak = xmax - xmin                   # 13. Peak-to-Peak
    xvar = np.var(signal)                        # 14. Variance 
    xmed = np.median(signal)                     # 15. Median
    mean_square = xrms**2                        # 16. Mean Square
    margin = xmax / np.sqrt(mean_square) if mean_square != 0 else 0  # 17. Margin
    impulse = xrms**2 / np.mean(np.abs(signal))  # 18. Impulse

    # 19. Median Frequency
    cumulative_power = np.cumsum(ps_norm)
    median_freq = freqs[np.searchsorted(cumulative_power, 0.5)]

    # 20. Frequency Center (same as mean frequency)
    freq_center = np.sum(freqs * ps_norm)

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

def extract_features_for_all_datasets(train_set, test_set, segment_size=1024):
    """Extract features from training and test datasets"""
    
    # Initialize dictionaries to store features
    train_features = {
        'raw': [], 'emd': [], 'eemd': [], 'ceemdan': [],
        'raw_emd': [], 'raw_eemd': [], 'raw_ceemdan': []
    }
    test_features = {
        'raw': [], 'emd': [], 'eemd': [], 'ceemdan': [],
        'raw_emd': [], 'raw_eemd': [], 'raw_ceemdan': []
    }
    
    # Process each fault condition
    for fault_type, train_signal in train_set.items():
        print(f"Processing fault type: {fault_type}")
        test_signal = test_set[fault_type]
        
        # Process training data
        train_segments = segment_signal(train_signal, frame_size=segment_size, overlap=0)
        train_segments = train_segments[:81]  # Ensure 81 segments per condition
        
        # Process test data
        test_segments = segment_signal(test_signal, frame_size=segment_size, overlap=0)
        test_segments = test_segments[:34]  # Ensure 34 segments per condition
        
        # Extract raw features
        train_raw_features = [extract_20_features(segment) for segment in train_segments]
        test_raw_features = [extract_20_features(segment) for segment in test_segments]
        
        # Add fault labels
        for features in train_raw_features:
            features['Fault_Type'] = fault_type
            train_features['raw'].append(features)
        
        for features in test_raw_features:
            features['Fault_Type'] = fault_type
            test_features['raw'].append(features)
        
        # Apply decomposition and extract features
        for decomp_type in ['emd', 'eemd', 'ceemdan']:
            print(f"Applying {decomp_type} decomposition...")
            train_decomp_features = []
            test_decomp_features = []
            
            for segment in train_segments:
                # Apply appropriate decomposition
                if decomp_type == 'emd':
                    imfs = emd(segment)
                elif decomp_type == 'eemd':
                    imfs = eemd(segment, emd_func=emd)
                else:  # ceemdan
                    imfs = ceemdan(segment, emd_func=emd)
                
                # Select most relevant IMFs and extract features
                selected_imfs = select_high_corr_imfs(segment, imfs)
                imf_features = extract_20_features(selected_imfs[0])
                imf_features['Fault_Type'] = fault_type
                train_decomp_features.append(imf_features)
            
            for segment in test_segments:
                if decomp_type == 'emd':
                    imfs = emd(segment)
                elif decomp_type == 'eemd':
                    imfs = eemd(segment, emd_func=emd)
                else:  # ceemdan
                    imfs = ceemdan(segment, emd_func=emd)
                
                selected_imfs = select_high_corr_imfs(segment, imfs)
                imf_features = extract_20_features(selected_imfs[0])
                imf_features['Fault_Type'] = fault_type
                test_decomp_features.append(imf_features)
            
            # Store decomposition features
            train_features[decomp_type].extend(train_decomp_features)
            test_features[decomp_type].extend(test_decomp_features)
            
            # Create combined feature sets (raw + decomposition)
            for i, raw_feat in enumerate(train_raw_features):
                combined = {**raw_feat, **{f"{decomp_type}_{k}": v 
                          for k, v in train_decomp_features[i].items() 
                          if k != 'Fault_Type'}}
                train_features[f'raw_{decomp_type}'].append(combined)
            
            for i, raw_feat in enumerate(test_raw_features):
                combined = {**raw_feat, **{f"{decomp_type}_{k}": v 
                          for k, v in test_decomp_features[i].items() 
                          if k != 'Fault_Type'}}
                test_features[f'raw_{decomp_type}'].append(combined)
    
    # Convert to DataFrames
    train_dfs = {k: pd.DataFrame(v) for k, v in train_features.items()}
    test_dfs = {k: pd.DataFrame(v) for k, v in test_features.items()}
    
    # Save feature files
    for k in train_dfs.keys():
        train_dfs[k].to_csv(f"{k}_train_features.csv", index=False)
        test_dfs[k].to_csv(f"{k}_test_features.csv", index=False)
    
    return train_dfs, test_dfs
