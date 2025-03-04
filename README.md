# Ewe Parturition

This Github Repository is divided into:
    
- ## isPartum/lightWeightModel/

    - This directory contains the [`transform_csv_in_parquet_5minsAggr.py`](./isPartum/lightWeightModel/transform_csv_in_parquet_5minsAggr.py) script which aggregates the data from the csv ewes files into 5 minutes intervals, using mean, median, max, min, std, and saves the result in a parquet file.

    - It also includes the [`rf_cop.py`](./isPartum/lightWeightModel/RF/rf_cop.py) script that trains a Random Forest lightweight model to predict the parturition of the ewes. 

        - **ISSUES**:
            - Performance of the model is not optimal (Low MCC) - check preprocessing steps and the aggregation process.

- ## isPartum/mccVsDeltaTimeVsSampleRate/RF

    - This directory contains the [`script.py`](./isPartum/mccVsDeltaTimeVsSampleRate/RF/script.py) script evaluates the impact of time window size and sampling rate on ewe parturition detection performance. It processes sensor data from several animals via complex feature engineering approaches (statistical, frequency-domain, and trend features) derived from accelerometer and temperature readings. The code uses `Optuna` to optimize the hyperparameters of `Random Forest` models over a range of time periods (15-180 minutes) and sampling rates (0.5-5Hz). To ensure `unbiased evaluation` - Instead of randomly splitting individual data points across all animals, the entire dataset for each individual animal is kept together and allocated entirely to either the training or testing set. This means if ewe #1234's data goes into the training set, all measurements from that specific animal stay in training, and none appear in testing - `data is separated at the animal level`, and class imbalance is handled by SMOTETomek resampling. Matthews Correlation Coefficient (`MCC`) is used to quantify performance, and the findings are presented as `2D line` plots and `3D surface` plots before being saved to CSV for interactive study. This extensive research aids in determining the best sensing settings for accurate parturition detection in sheep monitoring systems.