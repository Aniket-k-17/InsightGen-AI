# core/anomaly_detector.py
# Anomaly detection = finding rows that are UNUSUAL compared to the rest.
# Example: if everyone's salary is $50k-$80k, someone with $5,000,000 is an anomaly.
# We use 3 different methods so we catch more types of anomalies.

import numpy as np
import pandas as pd
from sklearn.ensemble    import IsolationForest    # method 1 - ML approach
from sklearn.preprocessing import StandardScaler   # used to prepare data for ML


def detect_anomalies(df):
    """
    Runs 3 anomaly detection methods on the DataFrame.
    Returns a dictionary with results from all 3 methods.
    Only works on NUMBER columns (text columns are skipped).
    """

    # Get only the number columns - anomaly detection needs numbers
    number_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # If there are no number columns at all, return an error message
    if not number_columns:
        return {"error": "No numeric columns found. Cannot run anomaly detection."}

    results = {}  # dictionary to store results from each method

    # ── Method 1: Isolation Forest ────────────────────────────────────────────
    # How it works: builds many random decision trees.
    # Anomalies are easier to "isolate" (separate) than normal data points.
    # contamination=0.05 means "expect about 5% of rows to be anomalies"

    # Step 1: Fill missing values with the median (the model needs complete data)
    clean_data = df[number_columns].fillna(df[number_columns].median())

    # Step 2: Scale the data (make all columns have similar ranges)
    # StandardScaler makes mean=0 and std=1 for each column
    # This prevents columns with big numbers (like salary=50000) from
    # dominating columns with small numbers (like age=25)
    scaler      = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    # Step 3: Train and predict
    model       = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(scaled_data)
    # fit_predict returns: +1 for normal rows, -1 for anomalies

    # Step 4: Create a mask (True/False list) - True = anomaly
    is_anomaly = predictions == -1

    results["isolation_forest_count"] = int(is_anomaly.sum())  # how many anomalies
    results["isolation_forest_pct"]   = round(is_anomaly.sum() / len(df) * 100, 2)
    results["anomaly_rows"]           = df[is_anomaly].head(10).reset_index(drop=True)

    # ── Method 2: Z-Score ─────────────────────────────────────────────────────
    # How it works: calculate how many "standard deviations" a value is from the mean.
    # Values more than 3 standard deviations away are considered outliers.
    # Z-score formula: z = (value - mean) / standard_deviation
    # If z > 3 or z < -3, the value is unusual.

    zscore_outliers = {}  # stores column_name: count_of_outliers

    for col in number_columns:
        std_dev = df[col].std()  # standard deviation of this column

        # Skip columns where all values are identical (std=0 would cause division by zero)
        if std_dev == 0:
            continue

        # Calculate z-score for every value in this column
        z_scores = (df[col] - df[col].mean()) / std_dev
        outlier_count = int((z_scores.abs() > 3).sum())

        if outlier_count > 0:
            zscore_outliers[col] = outlier_count

    results["zscore_outliers"] = zscore_outliers

    # ── Method 3: IQR (Interquartile Range) ──────────────────────────────────
    # How it works: this is the same rule used in box plots.
    # Q1 = 25th percentile (bottom quarter)
    # Q3 = 75th percentile (top quarter)
    # IQR = Q3 - Q1  (the middle 50% of data)
    # Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.

    iqr_outliers = {}  # stores column_name: {count, lower_bound, upper_bound}

    for col in number_columns:
        q1  = df[col].quantile(0.25)  # value at 25th percentile
        q3  = df[col].quantile(0.75)  # value at 75th percentile
        iqr = q3 - q1                 # interquartile range

        lower_fence = q1 - 1.5 * iqr  # anything below this is an outlier
        upper_fence = q3 + 1.5 * iqr  # anything above this is an outlier

        # Count how many values fall outside the fences
        outlier_count = int(((df[col] < lower_fence) | (df[col] > upper_fence)).sum())

        if outlier_count > 0:
            iqr_outliers[col] = {
                "count":       outlier_count,
                "lower_bound": round(lower_fence, 2),
                "upper_bound": round(upper_fence, 2),
            }

    results["iqr_outliers"] = iqr_outliers

    return results
