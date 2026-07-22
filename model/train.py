from __future__ import annotations

import os
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configuration
DEFAULT_DATA = Path("data/batches/month_01.csv")
EXPERIMENT = "fraud_detection"
REGISTERED_MODEL = "fraud_model"
# DB backend for the model registry.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SEED = 42

# Target
TARGET = "isFraud"

# Features
NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CATEGORICAL = ["type"]
FEATURES = NUMERIC + CATEGORICAL

def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a batch CSV and return (X = allowed features, y = fraud label)."""
    if not path.exists():
        raise SystemExit(
            f"\nBatch not found: {path}\n"
            "Generate the batches first:  python data/prepare_monthly_batches.py\n"
        )
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y

def build_pipeline(model) -> Pipeline:
    """Preprocess + classify in one object.

    - StandardScaler on the numeric columns
    - OneHotEncoder on `type`
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", model)])

def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the metrics that actually matter for rare-event fraud detection."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
    }
    
def setup_mlflow() -> None:
    """Point mlflow at the local SQLite store and keep artifacts in one folder."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        artifacts = Path("mlartifacts").resolve().as_uri()
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts)
    mlflow.set_experiment(EXPERIMENT)
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + track + register the fraud model.")
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Batch CSV to train on."
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_mlflow()

    X, y = load_features(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    print(
        f"Training on {args.data}  ({len(X_train):,} train / {len(X_test):,} test rows, "
        f"{y.sum():,} frauds total)"
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }

    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(model)
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_param("train_data", str(args.data))
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_test, pipe.predict(X_test))
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(3),
            )
            results[name] = {"metrics": metrics, "run_id": run.info.run_id}

    # Pick the better model by PR-AUC and register it as the deployable version.
    best = max(results, key=lambda n: results[n]["metrics"]["pr_auc"])
    best_run = results[best]["run_id"]
    model_version = mlflow.register_model(f"runs:/{best_run}/model", REGISTERED_MODEL)

    print_report(results, best, model_version)


def print_report(results: dict, best: str, model_version) -> None:
    cols = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_negatives",
    ]
    header = f"{'model':>20} " + " ".join(f"{c:>15}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["metrics"]
        row = f"{name:>20} " + " ".join(
            f"{m[c]:>15.4f}" if isinstance(m[c], float) else f"{m[c]:>15}" for c in cols
        )
        print(row)
    print(
        f"\nRegistered '{best}' as model '{REGISTERED_MODEL}' version "
        f"{model_version.version} (chosen by PR-AUC).\n"
        f"Inspect everything in the mlflow UI:\n"
        f"    mlflow ui --backend-store-uri {TRACKING_URI}\n"
        f"then open http://127.0.0.1:5000"
    )


if __name__ == "__main__":
    main()
