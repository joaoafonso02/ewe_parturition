import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


from imblearn.under_sampling import RandomUnderSampler


from joblib import parallel_backend


import tensorflow as tf
import tensorflow.keras.backend as K


import emlearn


import pickle


plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['lines.linewidth'] = 2

def preprocess_data(df):
    """Remove non-partum data and add features"""
    # Remove class 13
    df = df.filter(pl.col('Class') != 13)
    
    # Add features
    return df.with_columns([
        # Acceleration magnitude
        (pl.col('Acc_X (mg)').pow(2) + 
        pl.col('Acc_Y (mg)').pow(2) + 
        pl.col('Acc_Z (mg)').pow(2)).sqrt().alias('Acc_Magnitude'),
        
        # Rolling statistics
        pl.col('Acc_X (mg)').rolling_mean(window_size=5).alias('roll_mean_x'),
        pl.col('Acc_Y (mg)').rolling_mean(window_size=5).alias('roll_mean_y'),
        pl.col('Acc_Z (mg)').rolling_mean(window_size=5).alias('roll_mean_z')
    ])

def dataframe_shift(df, columns, windows):
    for i in range(1, windows):
        df = df.with_columns((pl.col(columns).shift(i)).name.prefix(f'prev_{i}_'))
    return df.drop_nulls()

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def get_mlp_keras(n_input, n_output):
    keras_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_input,)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(n_output, activation='softmax')
    ])
    
    keras_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[matthews_correlation_coefficient])

    return keras_mlp

def generate_models(n_input, n_output, light=False):
    # Remove MLP due to low performance and high computional resources
    # The code is present if anyone wants to explore further
    if light:
        models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(
            max_depth=128, max_features='sqrt', max_leaf_nodes=512, 
            random_state=42, class_weight='balanced')),
            ('RandomForestClassifier', RandomForestClassifier(n_estimators=32,
            max_depth=128, max_features='sqrt', max_leaf_nodes=512, 
            random_state=42, class_weight='balanced')),
            ('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=32,
            max_depth=128, max_features='sqrt', max_leaf_nodes=512, 
            random_state=42, class_weight='balanced')),
            #('Bagging', BaggingClassifier(n_estimators=32, max_features=0.1, random_state=42))
        ]
    else:
        models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42, class_weight='balanced')),
            ('RandomForestClassifier', RandomForestClassifier(random_state=42,  class_weight='balanced')),
            ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=42, class_weight='balanced')),
            ('Bagging', BaggingClassifier(random_state=42)),
            #('MLP (Keras)', get_mlp_keras(n_input, n_output))
        ]

    return models


def run_models(dfs, unique_labels, fr=None, n_components=0, light=False):
    rv = []
    for df_name, df in dfs:
        with parallel_backend('loky', n_jobs=-1):
            print(f'--- Dataset {df_name} ---')
            Y = df['Class'].to_numpy()
            X = df.drop(['Class', 'Time']).to_numpy()
            undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
            X, Y = undersample.fit_resample(X, Y)

            X_train, X_test, y_train, y_test = train_test_split(X, 
            Y, test_size=0.2, random_state=42, stratify=Y)

            # Feature Reduction
            if fr == 'PCA':
                pca = PCA(n_components=n_components)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
            elif fr == 'NMF':
                nmf = NMF(n_components=n_components, init='nndsvda', solver='mu', max_iter=1000)
                nmf.fit(X_train)
                X_train = nmf.transform(X_train)
                X_test = nmf.transform(X_test)
            
            best = None
            print(f'{"":<22} Accuracy Precision Recall F1-score   MCC')
            # Generate models
            num_classes = len(unique_labels)
            models = generate_models(X_train.shape[1], num_classes, light=light)
            for name, clf in models:
                if name == 'MLP (Keras)':
                    y_train_keras = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
                    y_test_keras = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

                    clf.fit(X_train, y_train_keras,
                    batch_size=2048,
                    epochs=500,
                    verbose=0,
                    validation_data=(X_test, y_test_keras))

                    y_pred = clf.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                print(f'{name:<22} {acc:>8.2f} {precision:>9.2f} {recall:>6.2f} {f1:>8.2f} {mcc:>5.2f}')

                if best is None:
                    best = (mcc, clf, name, y_test, y_pred)
                else:
                    if mcc > best[0]:
                        best = (mcc, clf, name, y_test, y_pred)
        _, clf, name, y_test, y_pred = best
        rv.append(clf)
        print(f'')
        print(f'Confusion Matrix for the best performing model: {name}')
        cf_matrix = confusion_matrix(y_test, y_pred)
        # print percentage 0-100 confusion matrix
        cm_percentage = np.nan_to_num((cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]) * 100, nan=0)
        print(cm_percentage)
        # save figure of the percentage
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'confusion_matrix_{df_name}_{name}.png')
        plt.close()

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix/np.sum(cf_matrix),
        display_labels = unique_labels)
        cm_display.plot(values_format='.2%', colorbar=False, cmap='Blues')
        plt.show()
        return rv
        
def load_directory_files(directory):
    """Load all CSV files from a directory"""
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if f.endswith('.csv')]

# Load files from train and test directories
train_files = load_directory_files('train')
test_files = load_directory_files('test')

print(f"Number of train files: {len(train_files)}")
print(f"Number of test files: {len(test_files)}")

# Process files function remains the same
def process_files(file_list):
    dfs = []
    for dataset in file_list:
        # Read CSV
        df = pl.read_csv(dataset, separator=';').with_columns(
            pl.col('Time').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%3f')
        )
        
        # Remove non-partum data and add features
        df = preprocess_data(df)
        
        # Scale accelerometer data
        df = df.with_columns([
            pl.col('Acc_X (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Y (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Z (mg)').map_batches(lambda x: pl.Series(minmax_scale(x)))
        ])
        
        # Resample to 1-second intervals
        df_resample = df.set_sorted('Time').group_by_dynamic('Time', every='1s').agg([
            pl.col('Acc_X (mg)').median(),
            pl.col('Acc_Y (mg)').median(),
            pl.col('Acc_Z (mg)').median(),
            pl.col('Temperature (C)').median(),
            pl.col('Class').mode().first()
        ])
        dfs.append(df_resample)
    return pl.concat(dfs)

# Process train and test sets
df_train = process_files(train_files)
df_test = process_files(test_files)



# PArtum classes are -1,0,1,2,3,4,5,6,7,8,9,10,11,12; Non-Partum is 13
# Plot class distribution
plt.figure(figsize=(15, 5))
classes = sorted(df_train['Class'].unique().to_list())
class_counts = df_train.group_by('Class').count()

plt.subplot(1, 2, 1)
plt.bar(class_counts['Class'], class_counts['count'])
plt.title('Hours until Partum Distribution')
plt.xlabel('Hours until Partum')
plt.ylabel('Number of Samples')

plt.subplot(1, 2, 2)
plt.pie(class_counts['count'], labels=class_counts['Class'], autopct='%1.1f%%')
plt.title('Class Distribution (%)')
plt.savefig('class_distribution.png')
plt.tight_layout()
plt.show()

# Prepare data for modeling
dfs = [
    ('Train', df_train),
    ('Test', df_test)
]

# Get unique labels for model training
unique_labels = sorted(df_train['Class'].unique().to_list())

# Train and evaluate models
models = run_models(dfs, unique_labels, light=True)

# reference models
print("\n-----------------Reference Models-----------------")
best_models = run_models([('AGG Median', df_train)], unique_labels, light=True)
# Store best model with EMLearn
cmodel = emlearn.convert(best_models[0], method='inline')
cmodel.save(file='../models/emlearn/clf_agg_median.h', name='clf_agg_median')
# Store model to use internally
with open('../models/pickle/clf_agg_median_light.pkl', 'wb') as f:
    pickle.dump(best_models[0], f, protocol=pickle.HIGHEST_PROTOCOL)

best_models = run_models([('AGG Median', df_train)], unique_labels, light=False)
# Store model to use internally
with open('../models/pickle/clf_agg_median.pkl', 'wb') as f:
    pickle.dump(best_models[0], f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n-----------------Sequence Learning-----------------")
window = 300 # 5min
df_roll_mdn = dataframe_shift(df_train, columns=['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)'], windows=window)

best_models = run_models([('Roll AGG Median', df_roll_mdn)], unique_labels, light=True)
# Store best model with EMLearn
cmodel = emlearn.convert(best_models[0], method='inline')
cmodel.save(file='../models/emlearn/clf_roll_agg_median.h', name='clf_roll_agg_median')
# Store model to use internally
with open('../models/pickle/clf_roll_agg_median_light.pkl', 'wb') as f:
    pickle.dump(best_models[0], f, protocol=pickle.HIGHEST_PROTOCOL)


best_models = run_models([('Roll AGG Median', df_roll_mdn)], unique_labels, light=False)
# Store model to use internally
with open('../models/pickle/clf_roll_agg_median.pkl', 'wb') as f:
    pickle.dump(best_models[0], f, protocol=pickle.HIGHEST_PROTOCOL)
