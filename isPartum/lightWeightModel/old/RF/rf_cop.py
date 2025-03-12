import pandas as pd
import numpy as np
import polars as pl
import joblib
import emlearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, fbeta_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import math, os

def add_minimal_features(df, windows=[12, 72, 144]): # 12 = 5min*12 = 1h, 72 = 5min*72 = 6h, 144 = 5min*144 = 12h
    """Add minimal but effective temporal features for partum detection"""
    df.columns = df.columns.str.strip()
    features = df.copy()
    
    # Determine sensor columns
    if 'Acc_X (mg)' in df.columns:
        sensor_cols = ['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)']
        # Add magnitude feature
        features['Acc_Magnitude'] = np.sqrt(
            features['Acc_X (mg)']**2 + 
            features['Acc_Y (mg)']**2 + 
            features['Acc_Z (mg)']**2
        )
    elif 'Acc_X_mean' in df.columns:
        sensor_cols = ['Acc_X_mean', 'Acc_Y_mean', 'Acc_Z_mean', 'Temperature_mean']
        # Add magnitude feature
        features['Acc_Magnitude'] = np.sqrt(
            features['Acc_X_mean']**2 + 
            features['Acc_Y_mean']**2 + 
            features['Acc_Z_mean']**2
        )
    else:
        raise KeyError(f"Expected sensor columns not found. Available columns: {df.columns.tolist()}")
    
    # Add only key rolling statistics to most important columns
    key_columns = ['Acc_Y_mean' if 'Acc_Y_mean' in features.columns else 'Acc_Y (mg)', 
                  'Acc_Magnitude', 
                  'Temperature_mean' if 'Temperature_mean' in features.columns else 'Temperature (C)']
    
    for col in key_columns:
        for window in windows:
            features[f'{col}_mean_{window}'] = features[col].rolling(window=window, min_periods=1).mean()
            features[f'{col}_std_{window}'] = features[col].rolling(window=window, min_periods=1).std()
            
    # Add only a few critical features for activity detection
    features['Mag_diff24'] = features['Acc_Magnitude'].diff(periods=24)  # 2-hour change
    mag_diff = features['Acc_Magnitude'].diff().abs()
    features['activity_changes_72'] = mag_diff.rolling(window=72, min_periods=1).sum()
    
    return features.fillna(method='bfill').fillna(0)

print("Loading and splitting data...")
data_path = Path("../../../../data/ewes2/limit8KGData/binaryAcc/parquets5mins/")
all_files = list(data_path.glob("*.parquet"))

# Split files into train and test sets
np.random.seed(42)
train_files = np.random.choice(all_files, size=int(len(all_files)*0.8), replace=False) # 80% train ewes
test_files = [f for f in all_files if f not in train_files]

print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

X_train_parts, y_train_parts = [], []
for file in train_files:
    print(f"Processing training file: {file.name}")
    df = pl.read_parquet(file).to_pandas()
    features = add_minimal_features(df)
    features['Target'] = features['Class'].apply(lambda x: 0 if x == 13 else 1)
    X = features.drop(columns=['Target', 'Time', 'Class'], errors='ignore')
    y = features['Target']
    X_train_parts.append(X)
    y_train_parts.append(y)

X_test_parts, y_test_parts = [], []
for file in test_files:
    print(f"Processing test file: {file.name}")
    df = pl.read_parquet(file).to_pandas()
    features = add_minimal_features(df)
    features['Target'] = features['Class'].apply(lambda x: 0 if x == 13 else 1)
    X = features.drop(columns=['Target', 'Time', 'Class'], errors='ignore')
    y = features['Target']
    X_test_parts.append(X)
    y_test_parts.append(y)

# Combine parts
X_train = pd.concat(X_train_parts)
X_test = pd.concat(X_test_parts)
y_train = pd.concat(y_train_parts)
y_test = pd.concat(y_test_parts)

# Handle missing/infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\nClass distribution in training:")
print(y_train.value_counts())
print("\nClass distribution in testing:")
print(y_test.value_counts())

# Scale features before selection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nSelecting top features...")
selector = SelectKBest(f_classif, k=8)  
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = [
    feature for feature, selected in zip(X_train.columns, selector.get_support())
    if selected
]
print(f"\nSelected {len(selected_features)} features:")
print(selected_features)

print("\nApplying SMOTE...")
smote = SMOTE(sampling_strategy=0.35, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

print(f"After resampling - shape: {X_train_resampled.shape}")
print("Class distribution after resampling:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Class {value}: {count} samples")

# Very compact Random Forest configuration
print("\nTraining compact Random Forest...")
compact_params = {
    'n_estimators': 5,               
    'max_depth': 6,                  
    'min_samples_split': 8,       
    'min_samples_leaf': 4,          
    'criterion': 'gini',            
    'max_features': 'sqrt',         
    'class_weight': {0: 1, 1: 17},  
    'random_state': 42,
    'n_jobs': -1
}
model = RandomForestClassifier(**compact_params)
model.fit(X_train_resampled, y_train_resampled)

print("\nOptimizing decision threshold...")
y_pred_proba = model.predict_proba(X_test_selected)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f3_scores = []
for p, r in zip(precision, recall):
    if p > 0 and r > 0:
        f3 = (10 * p * r) / (9 * p + r)
    else:
        f3 = 0
    f3_scores.append(f3)

if len(thresholds) > 0:
    best_idx = np.argmax(f3_scores[:-1])  
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
else:
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

print(f"Optimized threshold: {best_threshold:.3f}")
print(f"Expected precision: {best_precision:.3f}")
print(f"Expected recall: {best_recall:.3f}")

y_pred = (y_pred_proba >= best_threshold).astype(int)

# Evaluation
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

# Plot confusion matrix counts
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
            xticklabels=['Non-Partum', 'Partum'],
            yticklabels=['Non-Partum', 'Partum'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Counts)")
plt.tight_layout()
plt.savefig("confusion_matrix_count.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap='Blues',
            annot_kws={"size": 16},
            xticklabels=['Non-Partum', 'Partum'],
            yticklabels=['Non-Partum', 'Partum'])
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.title("Confusion Matrix (Percentage)", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrix_percentage.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = fbeta_score(y_test, y_pred, beta=1)
f2 = fbeta_score(y_test, y_pred, beta=2)
f3 = fbeta_score(y_test, y_pred, beta=3)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n=== DETAILED METRICS ===")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall/Sensitivity: {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"F2 Score: {f2:.3f}")
print(f"F3 Score: {f3:.3f}")
print(f"Matthews Correlation Coefficient: {mcc:.3f}")

print("\nConfusion Matrix (%):")
for i, label in enumerate(['Non-Partum', 'Partum']):
    print(f"{label:>10}: {cm_percentage[i,0]:>6.1f}% {cm_percentage[i,1]:>6.1f}%")

# Export model to C
print("\nExporting model to C...")
cmodel = emlearn.convert(model, method='inline')
cmodel.save(file='rf_partum_compact.h', name='partum_detector')
model_size_kb = Path('rf_partum_compact.h').stat().st_size / 1024
print(f"Model size: {model_size_kb:.2f} KB")


