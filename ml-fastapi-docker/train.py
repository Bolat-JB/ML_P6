from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

DROPPED_COLUMNS = ["fnlwgt", "education_num"]
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def load_data():
    """Fetch and clean the Adult Census Income dataset."""
    dataset = fetch_openml("adult", version=2, as_frame=True)
    X = dataset.data.copy()
    X.columns = X.columns.str.replace("-", "_", regex=False)
    X = X.drop(columns=DROPPED_COLUMNS)

    # Cast category columns to object before replacing '?' with NaN.
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype("object")
    X = X.replace("?", np.nan)

    y = dataset.target.astype(str).str.strip().map({"<=50K": 0, ">50K": 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels encountered while mapping y.")

    return X, y


def build_pipeline() -> Pipeline:
    """Create the preprocessing and classification pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = HistGradientBoostingClassifier(max_depth=8, random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved model to {MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
