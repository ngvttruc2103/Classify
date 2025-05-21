# main.py


from sklearn.model_selection import train_test_split
from Preprocessing import load_de_signals, remove_outliers, normalize
from Feature Extraction import extract_features_for_all_datasets

def prepare_datasets(folder_path):
    """Prepare the training and test datasets"""
    signals = load_de_signals(folder_path)
    
    train_set = {}
    test_set = {}
    
    for filename, signal in signals.items():
        print(f"Processing {filename}...")
        signal_cleaned = remove_outliers(signal)
        signals_normalized = normalize(signal_cleaned)
        
        X_train, X_test = train_test_split(signals_normalized, test_size=0.3, random_state=42)
        train_set[filename] = X_train
        test_set[filename] = X_test
        
    return train_set, test_set

def main():
    # Update this path to your CWRU data folder
    folder_path = r"E:\ON CLASS\Y4_S2\Thesis\Data\CWRU"
    
    print("Preparing datasets...")
    train_set, test_set = prepare_datasets(folder_path)
    
    print("Extracting features...")
    train_features, test_features = extract_features_for_all_datasets(train_set, test_set)
    
    print("Feature extraction completed!")
    if train_features:
        print("Available feature sets:", list(train_features.keys()))

if __name__ == "__main__":
    main()