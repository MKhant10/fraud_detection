
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

from auth import require_api_key

MODEL_NAME = "fraud_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
]
FRAUD_THRESHOLD = 0.5
REVIEW_MARGIN = 0.15
REVIEW_LOWER = FRAUD_THRESHOLD - REVIEW_MARGIN
REVIEW_UPPER = FRAUD_THRESHOLD + REVIEW_MARGIN
# MANUAL_FRAUD_THRESHOLD = 0.7

app = Flask(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
prediction_log = logging.getLogger("predictions")
prediction_log.setLevel(logging.INFO)
prediction_log.addHandler(logging.FileHandler(LOG_DIR / "predictions.log"))

# Load the model from MLflow
_model = None
_model_version = None

def load_model():
    """Fetch the latest registered model from mlflow and cache it in memory.

    Called at startup and by /reload, so the running service can pick up a newly
    retrained model version without a restart.
    """
    global _model, _model_version
    mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.sklearn.load_model(MODEL_URI)
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    _model_version = max((int(v.version) for v in versions), default=None)
    return _model_version

def classify(probability: float) -> str:
    """Map a fraud probability to a classification label."""
    if REVIEW_LOWER <= probability <= REVIEW_UPPER:
        return "review"
    return "fraud" if probability >= FRAUD_THRESHOLD else "legit"

#index
@app.get("/")
def home():
    return "Welcome to the Fraud Detection API. Use /health, /predict, or /reload endpoints."

@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
            "status": "ok",
            "model_loaded": _model is not None,
            "model_version": _model_version,
        })
    
@app.post("/predict")
@require_api_key
def predict():
    """Score a single application for fraud probability."""
    if _model is None:
        return (
            jsonify({"error": "No model loaded. Train and register a model first."}),
            503,
        )

    data = request.get_json(silent=True)
    print("Body data: ", data)
    if not isinstance(data, dict):
        return (
            jsonify({"error": "Send a JSON object with the application fields."}),
            400,
        )

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Build the one-row frame the model expects, then read the fraud probability.
    row = pd.DataFrame([{f: data[f] for f in FEATURES}])
    probability = float(_model.predict_proba(row)[0, 1])
    decision = classify(probability)

    result = {
        "fraud_probability": round(probability, 4),
        "decision" : decision,
        "needs_review" : decision == "review",
        "threshold": FRAUD_THRESHOLD,
        "is_fraud": probability >= FRAUD_THRESHOLD,
        "review_bound" : [REVIEW_LOWER, REVIEW_UPPER],
        "model_version": _model_version,
    }
    prediction_log.info(
        json.dumps({"time": datetime.now(timezone.utc).isoformat(), **result})
    )
    return jsonify(result)

@app.post("/reload")
@require_api_key
def reload():
    """Reload the latest registered model (call this right after a retrain)."""
    try:
        version = load_model()
        return jsonify({"status": "reloaded", "model_version": version})
    except Exception as exc:  # noqa: BLE001 - surface any load error to the caller
        return jsonify({"error": f"Reload failed: {exc}"}), 500
    
# Try to load a model at startup, but still start if the registry is empty
# (so /health works and the error is a clear 503 instead of a crash).
try:
    load_model()
except Exception as exc:  # noqa: BLE001
    app.logger.warning("Could not load a model at startup: %s", exc)
    

if __name__ == "__main__":
    app.run(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        debug=True,
    )