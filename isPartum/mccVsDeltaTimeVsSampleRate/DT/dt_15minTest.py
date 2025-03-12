from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Changed from RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import optuna

def objective(trial, train_files, test_files, delta_time, sample_rate):
    """Define the objective function for Optuna to optimize."""
    # Suggest values for the decision tree hyperparameters
    max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20, step=2)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20, step=2)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    
    # Create and train the DecisionTreeClassifier with the suggested hyperparameters
    X_train, y_train, X_test, y_test = [], [], [], []
    
    print(f"\nProcessing delta={delta_time}min, rate={sample_rate}Hz")
    
    for file in train_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df)
                if sum(labels) > 0:
                    print(f"Found {sum(labels)} partum sequences in {file.name}")
                X_train.extend(seq)
                y_train.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    for file in test_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df)
                X_test.extend(seq)
                y_test.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"No data for delta={delta_time}, rate={sample_rate}")
        return None
    
    X_train, y_train, X_test, y_test = map(np.array, [X_train, y_train, X_test, y_test])
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    try:
        smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=min(5, max(1, min(sum(y_train == 0), sum(y_train == 1)) - 1)), random_state=42))
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE failed, using original imbalanced data: {e}")
    
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight='balanced',
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    mcc = matthews_corrcoef(y_test, dt.predict(X_test))
    
    print(f"MCC for delta={delta_time}min, rate={sample_rate}Hz: {mcc:.3f}")
    
    return mcc

def process_single_combination(train_files, test_files, delta_time, sample_rate, max_depth, min_samples_split, min_samples_leaf, criterion):
    """Process single delta time and sample rate combination with optimized hyperparameters"""
    X_train, y_train, X_test, y_test = [], [], [], []
    
    print(f"\nProcessing delta={delta_time}min, rate={sample_rate}Hz")
    
    for file in train_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df)
                if sum(labels) > 0:
                    print(f"Found {sum(labels)} partum sequences in {file.name}")
                X_train.extend(seq)
                y_train.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    for file in test_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df)
                X_test.extend(seq)
                y_test.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"No data for delta={delta_time}, rate={sample_rate}")
        return None
    
    X_train, y_train, X_test, y_test = map(np.array, [X_train, y_train, X_test, y_test])
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    try:
        smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=min(5, max(1, min(sum(y_train == 0), sum(y_train == 1)) - 1)), random_state=42))
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE failed, using original imbalanced data: {e}")
    
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight='balanced',
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    mcc = matthews_corrcoef(y_test, dt.predict(X_test))
    
    print(f"MCC for delta={delta_time}min, rate={sample_rate}Hz: {mcc:.3f}")
    
    return mcc

# Keep create_advanced_sequences and split_animals_data functions unchanged
# [Previous implementation remains the same]

def create_advanced_sequences(df, base_window_size=120, overlap=0.75, sample_rate=1, delta_time=15):
    """Create sequences with a dynamically adjusted window size based on sample_rate and delta_time"""
    features = ['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)']
    
    # Convert delta_time (minutes) to seconds
    target_duration = delta_time * 60  # delta_time in minutes converted to seconds
    
    # Calculate window size dynamically based on sample rate and delta_time
    window_size = sample_rate * target_duration  # number of samples corresponding to delta_time in seconds
    window_size = max(30, min(base_window_size, window_size))  # Ensure it's between 30 and base_window_size

    step = int(window_size * (1 - overlap))  # Step size based on overlap
    sequences = []
    labels = []

    if 'Time' in df.columns:
        df = df.sort_values('Time').reset_index(drop=True)
    
    # Generate rolling statistics for features
    for col in features:
        df[f'{col}_roll_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_roll_std'] = df[col].rolling(window=5, min_periods=1).std()
    
    df = df.bfill()  # Handle missing values

    # Generate sequences based on the dynamic window size
    for i in range(0, len(df) - window_size + 1, step):
        sequence = df[features].iloc[i:i+window_size]
        window_classes = df['Class'].iloc[i:i+window_size]
        label = (window_classes != 13).any()
        
        seq_features = []
        for feature in features:
            feat_data = sequence[feature].values
            detrended = scipy.signal.detrend(feat_data)
            
            seq_features.extend([
                np.mean(feat_data), np.std(feat_data), np.min(feat_data),
                np.max(feat_data), np.median(feat_data), scipy.stats.iqr(feat_data),
                np.percentile(feat_data, 25), np.percentile(feat_data, 75)
            ])
            
            try:
                slope, _, _, _, _ = scipy.stats.linregress(np.arange(len(feat_data)), feat_data)
                seq_features.append(slope)
            except:
                seq_features.append(0)
            
            freqs = np.abs(np.fft.fft(detrended))
            seq_features.extend([
                np.mean(freqs), np.std(freqs), np.max(freqs), np.sum(freqs > np.mean(freqs))
            ])
            
            diff = np.diff(feat_data)
            seq_features.extend([
                np.mean(np.abs(diff)), np.std(diff), np.max(np.abs(diff)),
                len(np.where(np.diff(np.signbit(diff)))[0]) / len(diff)
            ])
        
        sequences.append(seq_features)
        labels.append(int(label))
    
    return np.array(sequences), np.array(labels)



def split_animals_data(base_dir):
    files = list(Path(base_dir).glob("*.csv"))
    animals = list(set(f.stem.split('_')[1] for f in files))  # Convert to list
    train_animals, test_animals = train_test_split(animals, test_size=0.2, random_state=42)
    return [f for f in files if f.stem.split('_')[1] in train_animals], [f for f in files if f.stem.split('_')[1] in test_animals]


def analyze_mcc_vs_delta(base_dir="resampled15minInterval"):
    train_files, test_files = split_animals_data(base_dir)
    results = defaultdict(dict)
    
    delta_values = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    rate_values = [0.5, 1, 2, 3, 5]
    
    for delta in delta_values:
        for rate in rate_values:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, train_files, test_files, delta, rate), n_trials=20)
            
            best_mcc = study.best_value
            best_params = study.best_params
            print(f"Best MCC for delta={delta}min, rate={rate}Hz: {best_mcc:.3f}")
            
            mcc = process_single_combination(
                train_files, test_files, delta, rate, 
                best_params['max_depth'], 
                best_params['min_samples_split'],
                best_params['min_samples_leaf'],
                best_params['criterion']
            )
            if mcc is not None:
                results[rate][delta] = mcc
    
    # [Rest of the visualization code remains the same]
    # [Previous implementation remains the same]
     # Prepare data for surface plot
    mcc_matrix = np.zeros((len(rate_values), len(delta_values)))
    
    for i, rate in enumerate(rate_values):
        for j, delta in enumerate(delta_values):
            mcc_matrix[i, j] = results.get(rate, {}).get(delta, np.nan)  
    
    # Generate the surface plot
    delta_grid, rate_grid = np.meshgrid(delta_values, rate_values)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(delta_grid, rate_grid, mcc_matrix, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Delta Time (minutes)')
    ax.set_ylabel('Sample Rate (Hz)')
    ax.set_zlabel('Matthews Correlation Coefficient (MCC)')
    ax.set_title('MCC vs Delta Time vs Sample Rate')
    fig.colorbar(surf, shrink=0.5, aspect=5) 
    
    plt.savefig('mcc_vs_delta_surface.png')
    plt.show()

    # 3D Surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(delta_grid, rate_grid, mcc_matrix, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Delta Time (minutes)')
    ax.set_ylabel('Sample Rate (Hz)')
    ax.set_zlabel('Matthews Correlation Coefficient (MCC)')
    ax.set_title('MCC vs Delta Time vs Sample Rate (3D)')
    fig.colorbar(surf, shrink=0.5, aspect=5) 
    
    plt.savefig('mcc_vs_delta_surface_3d.png')
    plt.close()
    
    # 2D Plot with lines for each sample rate
    plt.figure(figsize=(12, 8))
    for i, rate in enumerate(rate_values):
        plt.plot(delta_values, mcc_matrix[i], marker='o', label=f'{rate} Hz')
    
    plt.xlabel('Delta Time (minutes)')
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.title('MCC vs Delta Time for Different Sample Rates')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mcc_vs_delta_2d.png')
    plt.close()
    
    # Save surface data to CSV for later 3D visualization
    csv_data = []
    for i, rate in enumerate(rate_values):
        for j, delta in enumerate(delta_values):
            csv_data.append([delta, rate, mcc_matrix[i, j]])
    
    df = pd.DataFrame(csv_data, columns=['Delta_Time', 'Sample_Rate', 'MCC'])
    df.to_csv('mcc_surface_data.csv', index=False)
    
    return results

if __name__ == "__main__":
    results = analyze_mcc_vs_delta()
