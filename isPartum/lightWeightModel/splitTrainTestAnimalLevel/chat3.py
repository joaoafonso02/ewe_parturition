import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from joblib import parallel_backend
import emlearn

# Function to preprocess data
def preprocess_data(df):
    """Convert to binary classification and add features"""
    # Convert classes to binary (1 = Partum, 0 = Non-Partum)
    df = df.with_columns((pl.col('Class') != 13).cast(int).alias('Class'))
    
    return df.with_columns([
        # Acceleration magnitude
        (pl.col('Acc_X (mg)').pow(2) + 
        pl.col('Acc_Y (mg)').pow(2) + 
        pl.col('Acc_Z (mg)').pow(2)).sqrt().alias('Acc_Magnitude'),
        
        # Rolling statistics
        pl.col('Acc_X (mg)').rolling_mean(window_size=60).alias('roll_mean_x'),
        pl.col('Acc_Y (mg)').rolling_mean(window_size=60).alias('roll_mean_y'),
        pl.col('Acc_Z (mg)').rolling_mean(window_size=60).alias('roll_mean_z')
    ])

# Function to train and evaluate models
def run_models(dfs):
    best_model = None
    best_mcc = -1
    
    for df_name, df in dfs:
        with parallel_backend('loky', n_jobs=-1):
            print(f'--- Dataset {df_name} ---')
            Y = df['Class'].to_numpy()
            X = df.drop(['Class', 'Time']).to_numpy()

            print(f'Class distribution before undersampling: {np.unique(Y, return_counts=True)}')
            
            # Handle class imbalance
            undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
            X, Y = undersample.fit_resample(X, Y)

            print(f'Class distribution after undersampling: {np.unique(Y, return_counts=True)}')
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            
            # Train models
            models = [
                ('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')),
                ('RandomForestClassifier', RandomForestClassifier(n_estimators=32, max_depth=10, random_state=42, class_weight='balanced'))
            ]
            
            print(f'{"":<22} Accuracy Precision Recall F1-score   MCC')
            for name, clf in models:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                mcc = metrics.matthews_corrcoef(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                
                print(f'{name:<22} {acc:>8.2f} {precision:>9.2f} {recall:>6.2f} {f1:>8.2f} {mcc:>5.2f}')
                
                print(f'Confusion Matrix for: {name}')
                cf_matrix = confusion_matrix(y_test, y_pred)
                cm_percentage = np.nan_to_num((cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]) * 100, nan=0)
                print(cm_percentage)

                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Percentage'})
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for: {name}')
                plt.tight_layout()
                plt.savefig(f'confusion_matrix_{name}.png')
                plt.show()
                
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_model = clf   

            print(f'Best Model: {best_model}')
            
    return best_model

# Function to load CSV files
def load_directory_files(directory):
    """Load all CSV files from a directory"""
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if f.endswith('.csv')]

# Load and process data
train_files = load_directory_files('train')
test_files = load_directory_files('test')

print(f'Train files:', len(train_files))
print(f'Test files:', len(test_files))

def process_files(file_list):
    dfs = []
    for dataset in file_list:
        df = pl.read_csv(dataset, separator=';').with_columns(
            pl.col('Time').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%3f')
        )
        
        # Scale accelerometer data - use this approach instead
        acc_x = df['Acc_X (mg)'].to_numpy()
        acc_y = df['Acc_Y (mg)'].to_numpy()
        acc_z = df['Acc_Z (mg)'].to_numpy()
        
        df = df.with_columns([
            pl.lit(minmax_scale(acc_x)).alias('Acc_X (mg)'),
            pl.lit(minmax_scale(acc_y)).alias('Acc_Y (mg)'),
            pl.lit(minmax_scale(acc_z)).alias('Acc_Z (mg)')
        ])
        
        # Resample to 1-second intervals
        df = df.set_sorted('Time').group_by_dynamic('Time', every='1s').agg([
            pl.col('Acc_X (mg)').median(),
            pl.col('Acc_Y (mg)').median(),
            pl.col('Acc_Z (mg)').median(),
            pl.col('Temperature (C)').median(),
            pl.col('Class').mode().first()
        ])
        
        # Apply other preprocessing (now on resampled data)
        df = preprocess_data(df)
        dfs.append(df)
    
    return pl.concat(dfs)

# Process train and test sets
df_train = process_files(train_files)
df_test = process_files(test_files)

print("Files processed")

# Prepare datasets
dfs = [
    ('Train', df_train),
    ('Test', df_test)
]

# Train and evaluate models
best_model = run_models(dfs)

# Store best model with EMLearn
cmodel = emlearn.convert(best_model, method='inline')
cmodel.save(file='../models/emlearn/isPartum_lightweight.h', name='binaryClassificator')
