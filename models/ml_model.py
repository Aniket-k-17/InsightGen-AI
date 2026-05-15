# models/ml_model.py
# All Machine Learning logic in one simple file.
# We use scikit-learn library which makes ML very easy in Python.

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
    Clean and prepare data before training.
    - Removes rows with missing values
    - Converts text columns to numbers (ML only understands numbers)
    """
    # Keep only the columns we need, drop rows with missing values
    data = df[feature_columns + [target_column]].dropna().copy()

    X = data[feature_columns].copy()   # inputs
    y = data[target_column].copy()     # output (what to predict)

    # Convert any text column to numbers
    # Example: ["cat", "dog", "cat"] becomes [0, 1, 0]
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col].astype(str))
            encoders[col] = enc

    # Also convert target if it is text
    if y.dtype == "object":
        enc = LabelEncoder()
        y = pd.Series(enc.fit_transform(y.astype(str)))
        encoders["__target__"] = enc

    return X, y, encoders


def get_problem_type(df, target_column):
    """
    Decides if this is Regression or Classification.
    - Text target          → Classification
    - Numbers, few unique  → Classification
    - Numbers, many unique → Regression
    """
    target = df[target_column].dropna()
    if target.dtype == "object":
        return "classification"
    if target.nunique() <= 10:
        return "classification"
    return "regression"


def train_model(df, feature_columns, target_column, model_name, test_size=0.2):
    """
    Main function: prepares data, trains model, returns results dict.
    """
    problem_type = get_problem_type(df, target_column)

    # Prepare the data
    X, y, encoders = prepare_data(df, feature_columns, target_column)

    # Split: 80% train (model learns), 20% test (we check accuracy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Pick the right model
    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(max_depth=5),
            "Random Forest":     RandomForestRegressor(n_estimators=100),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(max_depth=5),
            "Random Forest":       RandomForestClassifier(n_estimators=100),
        }

    model = models[model_name]
    model.fit(X_train, y_train)   # train
    y_pred = model.predict(X_test) # predict

    # Score
    if problem_type == "regression":
        score = round(r2_score(y_test, y_pred) * 100, 1)
        error = round(mean_absolute_error(y_test, y_pred), 3)
    else:
        score = round(accuracy_score(y_test, y_pred) * 100, 1)
        error = None

    # Feature importance
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature":    feature_columns,
            "Importance": model.feature_importances_.round(4),
        }).sort_values("Importance", ascending=False)

    return {
        "model":        model,
        "problem_type": problem_type,
        "encoders":     encoders,
        "score":        score,
        "error":        error,
        "y_test":       list(y_test),
        "y_pred":       list(y_pred),
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
        "importance":   importance,
    }


def predict_single(result, feature_columns, user_inputs):
    """
    Makes one prediction from values the user typed in manually.
    """
    row = pd.DataFrame([user_inputs])

    # Apply same encoding used during training
    for col in row.columns:
        if col in result["encoders"]:
            row[col] = result["encoders"][col].transform(row[col].astype(str))

    prediction = result["model"].predict(row)[0]

    # Decode back to original label if target was text
    if "__target__" in result["encoders"]:
        prediction = result["encoders"]["__target__"].inverse_transform([int(prediction)])[0]

    return prediction