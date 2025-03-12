from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats
import scipy.signal
import optuna

def create_advanced_sequences(df, base_window_size=120, overlap=0.75, sample_rate=1, delta_time=15):
    features = ['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)']
    target_duration = delta_time * 60
    window_size = max(30, min(base_window_size, sample_rate * target_duration))
    step = int(window_size * (1 - overlap))
    sequences, labels = [], []

    if 'Time' in df.columns:
        df = df.sort_values('Time').reset_index(drop=True)
    
    for col in features:
        df[f'{col}_roll_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_roll_std'] = df[col].rolling(window=5, min_periods=1).std()
    
    df = df.bfill()

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

def objective_rf(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 50, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=2),
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return matthews_corrcoef(y_test, model.predict(X_test))

def objective_dt(trial, X_train, y_train, X_test, y_test):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=2),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return matthews_corrcoef(y_test, model.predict(X_test))

def objective_knn(trial, X_train, y_train, X_test, y_test):
   # Get minimum possible n_neighbors based on sample size
    min_samples = min(len(X_train), len(X_test))
    max_neighbors = min(15, min_samples - 1)  # Ensure n_neighbors < n_samples
    
    # Adjust n_neighbors range based on available samples
    if max_neighbors < 3:  # Too few samples for KNN
        return 0.0
        
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, max_neighbors),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        'n_jobs': -1
    }
    
    try:
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        return matthews_corrcoef(y_test, model.predict(X_test))
    except Exception as e:
        print(f"KNN failed: {e}")
        return 0.0

def prepare_data(train_files, test_files, delta_time, sample_rate):
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for file in train_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df, sample_rate=sample_rate, delta_time=delta_time)
                X_train.extend(seq)
                y_train.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    for file in test_files:
        if f"_{sample_rate}Hz_{delta_time}min.csv" in file.name:
            try:
                df = pd.read_csv(file, sep=';').dropna()
                seq, labels = create_advanced_sequences(df, sample_rate=sample_rate, delta_time=delta_time)
                X_test.extend(seq)
                y_test.extend(labels)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
                
    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None
        
    X_train, y_train, X_test, y_test = map(np.array, [X_train, y_train, X_test, y_test])
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    try:
        # Adjust SMOTE k_neighbors based on sample size
        n_samples = len(X_train)
        k_neighbors = min(5, max(1, min(sum(y_train == 0), sum(y_train == 1)) - 1))
        
        if k_neighbors > 0 and n_samples > k_neighbors:
            smote_tomek = SMOTETomek(
                random_state=42,
                smote=SMOTE(k_neighbors=k_neighbors, random_state=42)
            )
            X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE failed: {e}")
        
    return X_train, y_train, X_test, y_test

def run_model_comparison(train_files, test_files, delta_time, sample_rate, model_type='rf'):
    X_train, y_train, X_test, y_test = prepare_data(train_files, test_files, delta_time, sample_rate)
    
    if X_train is None:
        return None
        
    study = optuna.create_study(direction="maximize")
    n_trials = 20
    
    if model_type == 'rf':
        objective = lambda trial: objective_rf(trial, X_train, y_train, X_test, y_test)
    elif model_type == 'dt':
        objective = lambda trial: objective_dt(trial, X_train, y_train, X_test, y_test)
    else:  # knn
        objective = lambda trial: objective_knn(trial, X_train, y_train, X_test, y_test)
    
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_mcc = study.best_value
    
    print(f"\nBest parameters for {model_type.upper()} (delta={delta_time}, rate={sample_rate}):")
    print(best_params)
    print(f"Best MCC: {best_mcc:.3f}")
    
    return best_mcc

def split_animals_data(base_dir):
    files = list(Path(base_dir).glob("*.csv"))
    animals = list(set(f.stem.split('_')[1] for f in files))
    train_animals, test_animals = train_test_split(animals, test_size=0.2, random_state=42)
    return [f for f in files if f.stem.split('_')[1] in train_animals], [f for f in files if f.stem.split('_')[1] in test_animals]

def compare_models(base_dir="resampled15minInterval"):
    train_files, test_files = split_animals_data(base_dir)
    results = {
        'rf': defaultdict(dict),
        'dt': defaultdict(dict),
        'knn': defaultdict(dict)
    }
    
    delta_values = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    rate_values = [0.5, 1, 2, 3, 5]
    
    for model_type in ['rf', 'dt', 'knn']:
        print(f"\nOptimizing {model_type.upper()} model")
        for delta in delta_values:
            for rate in rate_values:
                mcc = run_model_comparison(train_files, test_files, delta, rate, model_type)
                if mcc is not None:
                    results[model_type][rate][delta] = mcc
    
    # Visualization
    fig = plt.figure(figsize=(20, 6))
    
    for idx, (model_type, title) in enumerate([('rf', 'Random Forest'), 
                                             ('dt', 'Decision Tree'), 
                                             ('knn', 'K-Nearest Neighbors')]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        mcc_matrix = np.zeros((len(rate_values), len(delta_values)))
        for i, rate in enumerate(rate_values):
            for j, delta in enumerate(delta_values):
                mcc_matrix[i, j] = results[model_type].get(rate, {}).get(delta, np.nan)
        
        delta_grid, rate_grid = np.meshgrid(delta_values, rate_values)
        surf = ax.plot_surface(delta_grid, rate_grid, mcc_matrix, cmap='viridis')
        
        ax.set_xlabel('Delta Time (min)')
        ax.set_ylabel('Sample Rate (Hz)')
        ax.set_zlabel('MCC')
        ax.set_title(title)
        plt.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('model_comparison_3d.png')
    plt.close()
    
    # 2D comparison plot
    plt.figure(figsize=(15, 8))
    markers = {'rf': 'o', 'dt': 's', 'knn': '^'}
    colors = {'rf': 'blue', 'dt': 'green', 'knn': 'red'}
    labels = {'rf': 'Random Forest', 'dt': 'Decision Tree', 'knn': 'KNN'}
    
    for model_type in results:
        for rate in rate_values:
            mcc_values = [results[model_type][rate].get(delta, np.nan) for delta in delta_values]
            plt.plot(delta_values, mcc_values, 
                    marker=markers[model_type],
                    color=colors[model_type],
                    alpha=0.6,
                    label=f'{labels[model_type]} ({rate}Hz)')
    
    plt.xlabel('Delta Time (minutes)')
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.title('Model Comparison: MCC vs Delta Time for Different Sample Rates')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_2d.png')
    plt.close()
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Model': model_type.upper(),
            'Delta_Time': delta,
            'Sample_Rate': rate,
            'MCC': results[model_type][rate].get(delta, np.nan)
        }
        for model_type in results
        for rate in rate_values
        for delta in delta_values
    ])
    
    results_df.to_csv('model_comparison_results.csv', index=False)
    return results

if __name__ == "__main__":
    results = compare_models()
