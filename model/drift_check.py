from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASELINE = Path("data/batches/month_01.csv")
DRIFT_THRESHOLD = 0.25  # PSI above this on any feature = significant drift

NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CATEGORICAL = ["type"]


def psi_numeric(
    baseline: np.ndarray, new: np.ndarray, bins: int = 10, eps: float = 1e-4
) -> float:
    """PSI for a numeric feature. Bin edges come from the baseline's quantiles,
    so we compare like-for-like regions of the distribution."""
    baseline = np.asarray(baseline, dtype=float)
    new = np.asarray(new, dtype=float)
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0  # baseline is constant -> no meaningful bins
    interior = edges[1:-1]
    n_bins = len(edges) - 1
    base_counts = np.bincount(np.digitize(baseline, interior), minlength=n_bins).astype(
        float
    )
    new_counts = np.bincount(np.digitize(new, interior), minlength=n_bins).astype(float)
    base_prop = np.clip(base_counts / base_counts.sum(), eps, None)
    new_prop = np.clip(new_counts / new_counts.sum(), eps, None)
    return float(np.sum((new_prop - base_prop) * np.log(new_prop / base_prop)))


def psi_categorical(baseline: pd.Series, new: pd.Series, eps: float = 1e-4) -> float:
    """PSI for a categorical feature. Buckets are the categories themselves."""
    base_prop = baseline.value_counts(normalize=True)
    new_prop = new.value_counts(normalize=True)
    categories = set(base_prop.index) | set(new_prop.index)
    psi = 0.0
    for c in categories:
        b = max(float(base_prop.get(c, 0.0)), eps)
        n = max(float(new_prop.get(c, 0.0)), eps)
        psi += (n - b) * np.log(n / b)
    return float(psi)


def compute_drift(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """Compute PSI per feature and decide whether the batch has drifted.

    Returns a dict with the per-feature PSI, the worst feature, and a boolean
    `drifted` flag (True if ANY monitored feature exceeds the threshold).
    """
    per_feature = {}
    for f in NUMERIC:
        per_feature[f] = psi_numeric(baseline_df[f].values, batch_df[f].values)
    for f in CATEGORICAL:
        per_feature[f] = psi_categorical(baseline_df[f], batch_df[f])

    triggered = [f for f, v in per_feature.items() if v > threshold]
    worst = max(per_feature, key=per_feature.get)
    return {
        "per_feature": per_feature,
        "worst_feature": worst,
        "max_psi": per_feature[worst],
        "threshold": threshold,
        "drifted": bool(triggered),
        "triggered": triggered,
    }


def _label(psi: float) -> str:
    if psi > 0.25:
        return "SIGNIFICANT"
    if psi > 0.10:
        return "moderate"
    return "stable"


def print_report(result: dict, baseline_path: Path, batch_path: Path) -> None:
    print(f"\nBaseline: {baseline_path}")
    print(f"Batch:    {batch_path}")
    print(f"\n{'feature':>16} {'PSI':>10}   status")
    print("-" * 42)
    for f, v in result["per_feature"].items():
        print(f"{f:>16} {v:>10.4f}   {_label(v)}")
    verdict = "DRIFT DETECTED" if result["drifted"] else "no drift"
    print("-" * 42)
    print(f"threshold = {result['threshold']}  ->  {verdict}")
    if result["drifted"]:
        print(f"triggered by: {', '.join(result['triggered'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check a batch for data drift (PSI).")
    p.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="CSV the current model was trained on.",
    )
    p.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Incoming batch CSV to check for drift.",
    )
    p.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.baseline, args.batch):
        if not path.exists():
            raise SystemExit(f"\nFile not found: {path}\n")
    baseline_df = pd.read_csv(args.baseline)
    batch_df = pd.read_csv(args.batch)
    result = compute_drift(baseline_df, batch_df, threshold=args.threshold)
    print_report(result, args.baseline, args.batch)


if __name__ == "__main__":
    main()