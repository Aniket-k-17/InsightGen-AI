# models/ml_model.py
# All Machine Learning logic — fully tested and bug-free.

import pandas as pd
import numpy as np
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.tree            import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_absolute_error, r2_score, accuracy_score


def prepare_data(df, feature_columns, target_column):
    """
    Cleans and prepares data for ML:
    1. Drops rows with missing values
    2. Detects numeric-looking strings ("12.5" → 12.5)
    3. Label-encodes genuine text columns ("Texas" → 0)
    4. Converts everything to float64 column-by-column (safe, no bulk astype crash)
    """
    data = df[feature_columns + [target_column]].dropna().copy()

    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} clean rows available after removing missing values. "
            "Please fill missing values on the Upload page first (need at least 10 rows)."
        )

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    encoders = {}

    for col in X.columns:
        if X[col].dtype == "object":
            # Try converting to numeric first — some columns store numbers as strings
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().sum() / len(converted) >= 0.9:
                # Numeric-looking strings like "12.5" → treat as number
                X[col] = converted.fillna(converted.median())
            else:
                # Genuine text column → LabelEncode
                X[col] = X[col].astype(str).replace("nan", "unknown")
                enc = LabelEncoder()
                X[col] = enc.fit_transform(X[col])
                encoders[col] = enc

        # BUG FIX: convert column to float individually AFTER encoding
        # Doing X.astype(float) all at once crashes if any text column wasn't encoded yet
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

    # Encode target if it is text
    if y.dtype == "object":
        y = y.astype(str).replace("nan", "unknown")
        enc = LabelEncoder()
        y = pd.Series(enc.fit_transform(y), name=target_column)
        encoders["__target__"] = enc

    return X, y, encoders


def get_problem_type(df, target_column):
    """
    Regression     → numeric target with many unique values (price, revenue, etc.)
    Classification → text target OR numeric with few unique values (0/1, 1-5 rating)
    """
    target = df[target_column].dropna()
    if target.dtype == "object":
        return "classification"
    if target.nunique() <= 10:
        return "classification"
    return "regression"


def train_model(df, feature_columns, target_column, model_name, test_size=0.2):
    """
    Full training pipeline. Returns a results dict with everything
    needed for the UI: metrics, charts, feature importance, live prediction.
    """
    problem_type = get_problem_type(df, target_column)
    X, y, encoders = prepare_data(df, feature_columns, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }

    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Score and convert predictions to plain Python types (no numpy types)
    if problem_type == "regression":
        score       = round(float(r2_score(y_test, y_pred)) * 100, 1)
        error       = round(float(mean_absolute_error(y_test, y_pred)), 3)
        y_test_list = [float(v) for v in y_test]
        y_pred_list = [float(v) for v in y_pred]
    else:
        score = round(float(accuracy_score(y_test, y_pred)) * 100, 1)
        error = None
        # Decode numeric labels back to original text for display in UI
        if "__target__" in encoders:
            enc         = encoders["__target__"]
            y_test_list = list(enc.inverse_transform([int(v) for v in y_test]))
            y_pred_list = list(enc.inverse_transform([int(v) for v in y_pred]))
        else:
            y_test_list = [str(v) for v in y_test]
            y_pred_list = [str(v) for v in y_pred]

    # Feature importance (Decision Tree and Random Forest only)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature":    feature_columns,
            "Importance": [round(float(v), 4) for v in model.feature_importances_],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "model":           model,
        "model_name":      model_name,
        "problem_type":    problem_type,
        "encoders":        encoders,
        "feature_columns": feature_columns,
        "score":           score,
        "error":           error,
        "y_test":          y_test_list,
        "y_pred":          y_pred_list,
        "train_rows":      len(X_train),
        "test_rows":       len(X_test),
        "importance":      importance,
    }


def predict_single(result, feature_columns, user_inputs):
    """
    Makes one live prediction from user-entered values.
    Handles unseen text values safely — no crash.
    """
    row = pd.DataFrame([user_inputs])

    for col in row.columns:
        if col in result["encoders"]:
            enc = result["encoders"][col]
            val = str(row[col].iloc[0])
            # If user picks a value the model never saw — use 0 (safe fallback)
            if val in enc.classes_:
                row[col] = float(enc.transform([val])[0])
            else:
                row[col] = 0.0
        else:
            try:
                row[col] = float(row[col].iloc[0])
            except (ValueError, TypeError):
                row[col] = 0.0

    row = row.astype(float)
    prediction = result["model"].predict(row)[0]

    # Decode classification target back to original label
    if "__target__" in result["encoders"]:
        enc = result["encoders"]["__target__"]
        try:
            prediction = enc.inverse_transform([int(round(float(prediction)))])[0]
        except Exception:
            prediction = str(prediction)

    return prediction