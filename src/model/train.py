from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
import numpy as np

def prepare_training_data(df):
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    X = df.drop(columns=string_cols + [
        "target","pts","pts_opp","won",
        "usg%_roll10","usg%_opp_roll10","usg%_roll10_opp_history","mp"
    ])
    y = df["target"]

    return X, y


def time_split(X, y, test_size=0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        shuffle=False
    )


def cross_validate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    return np.mean(scores)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)

    return model, val_accuracy

def retrain_full(model, X, y):
    model.fit(X, y)
    return model

import joblib

def save_model(model):
    path = "model_pipeline.pkl"
    joblib.dump(model, path)

def train_pipeline(df):

    X, y = prepare_training_data(df)

    X_train, X_val, y_train, y_val = time_split(X, y)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, C=0.01, tol=0.001)
    )


    cv_score = cross_validate_model(model, X_train, y_train)

    model, val_accuracy = train_and_evaluate(
        model,
        X_train, y_train,
        X_val, y_val
    )

    print(f"Cross-val accuracy: {cv_score:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    model_pipeline = retrain_full(model, X, y)
    save_model(model_pipeline)

    return val_accuracy