"""
Retraining automation: walk through the 12 simulated months and, for each one,
decide whether to retrain the model. This is the heart of the MLOps loop.

RETRAIN TRIGGERS (whichever comes first)
    - drift:    any monitored feature's PSI vs the baseline exceeds the threshold
    - schedule: a fixed cadence (every 3rd month), like a monthly/quarterly cron

DESIGN CHOICES
    - The drift baseline is FIXED at month 1 - the reference the model was first
      trained and validated on. We compare every incoming month to this known-
      good reference, not to a moving target: a moving baseline can hide slow,
      steady drift because it creeps along with the data and never trips.
    - When a retrain fires we retrain on that month's batch (the freshest data)
      by calling the training script - the same script the GitHub Action runs.
      A production system might retrain on an accumulating window; a single month
      keeps the demo simple and fast.
    - After each retrain we POST to the running API's /reload endpoint so the
      live service immediately serves the new version - the loop is connected,
      not two disconnected halves. If the API is not running the step is skipped.

Run from the repo root:
    python automation/retrain.py
"""

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from drift_check import compute_drift

BATCH_DIR = ROOT / "data" / "batches"
BASELINE = BATCH_DIR / "month_01.csv"
N_MONTHS = 12
SCHEDULE_EVERY = 3  # a scheduled retrain every 3rd month (quarterly cron)
DRIFT_THRESHOLD = 0.25
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "fraud_model"

# The running API to refresh after each retrain (the "redeploy" step). If it is
# not running (e.g. in CI), the reload is skipped gracefully.
load_dotenv()
API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("FRAUD_API_KEY", "")


def latest_version():
    """Highest registered version number of the fraud model (or None)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    return max((int(v.version) for v in versions), default=None)


def retrain_on(batch_path: Path):
    """Retrain by calling the training script on a batch, which registers a new
    model version. Using the real script (not an import) mirrors exactly what the
    scheduled GitHub Action does."""
    subprocess.run(
        [sys.executable, str(ROOT / "model" / "train.py"), "--data", str(batch_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return latest_version()


def reload_api() -> str:
    """Tell the running API to load the newest model - the redeploy step that
    keeps the served model in sync with the registry. Skipped if the API is not
    running, so the simulation and CI still work."""
    try:
        resp = requests.post(
            f"{API_URL}/reload", headers={"X-API-Key": API_KEY}, timeout=10
        )
        if resp.status_code == 200:
            return f"API now serving v{resp.json().get('model_version')}"
        return f"API reload failed (HTTP {resp.status_code})"
    except requests.exceptions.RequestException:
        return "API not running - reload skipped"


def decide(month: int, drift: dict) -> tuple[bool, str]:
    """Return (should_retrain, reason) for a given month."""
    reasons = []
    if drift["drifted"]:
        reasons.append("drift(" + ",".join(drift["triggered"]) + ")")
    if month % SCHEDULE_EVERY == 0:
        reasons.append("schedule")
    return (len(reasons) > 0), " + ".join(reasons)


def main() -> None:
    baseline_df = pd.read_csv(BASELINE)

    print(f"\n{'month':>5} {'max_psi':>8} {'drift?':>7} {'scheduled?':>10}   action")
    print("-" * 58)
    print(f"{1:>5} {'-':>8} {'-':>7} {'-':>10}   initial model (v{latest_version()})")

    for month in range(2, N_MONTHS + 1):
        batch_df = pd.read_csv(BATCH_DIR / f"month_{month:02d}.csv")
        drift = compute_drift(baseline_df, batch_df, threshold=DRIFT_THRESHOLD)
        should_retrain, reason = decide(month, drift)
        scheduled = "YES" if month % SCHEDULE_EVERY == 0 else "no"

        if should_retrain:
            version = retrain_on(BATCH_DIR / f"month_{month:02d}.csv")
            action = f"RETRAIN -> v{version}  [{reason}]  ({reload_api()})"
        else:
            action = "skip"

        print(
            f"{month:>5} {drift['max_psi']:>8.3f} "
            f"{('YES' if drift['drifted'] else 'no'):>7} {scheduled:>10}   {action}"
        )

    print(f"\nFinal registered version: {latest_version()}")


if __name__ == "__main__":
    main()
