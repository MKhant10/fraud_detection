
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Default values
DEFAULT_RAW = Path("data/raw/paysim.csv")
DEFAULT_OUT = Path("data/batches")
N_MONTHS = 12
DEFAULT_SUBSAMPLE = 1_000_000
DEFAULT_SEED = 42

EXPECTED_COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Drift plan: it is placed between the quarterly scheduled retrains
DRIFT_PLAN = {
    5: {"kind": "amount_inflation", "factor": 2.0},  # "transactions got bigger"
    8: {"kind": "type_shift", "to_transfer_frac": 0.30},  # "more money via TRANSFER"
    11: {"kind": "amount_inflation", "factor": 3.0},  # a stronger repeat, later on
}


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw PaySim CSV, failing with a helpful message if it is missing
    or does not have the columns we expect."""
    if not path.exists():
        raise SystemExit(
            f"\nRaw dataset not found at: {path}\n"
            "Download PaySim from https://www.kaggle.com/datasets/ealaxi/paysim1/data,\n"
            "rename the CSV to 'paysim.csv', and place it in data/raw/ (see README).\n"
        )
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"\nThe CSV at {path} is missing expected PaySim columns: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
        )
    return df


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Shrink the data to ~n rows while KEEPING the true fraud rate.

    We sample the same fraction from the fraud and non-fraud groups separately
    (stratified), so the natural ~0.13% imbalance is preserved. That keeps the
    imbalance story honest instead of accidentally making fraud look common.
    """
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    out = df.groupby("isFraud", group_keys=False).sample(frac=frac, random_state=seed)
    return out.reset_index(drop=True)


def split_into_months(df: pd.DataFrame, n_months: int, seed: int) -> list[pd.DataFrame]:
    """Shuffle, then cut into n_months equal slices.

    We split RANDOMLY (i.i.d.) rather than by PaySim's `step` timeline on purpose.
    Reason: we want every "clean" month to be statistically identical, so that
    when we inject drift into a few months it stands out as the ONLY change -
    not confounded by PaySim's natural fraud-rate swings across its 30 days. The
    12 months are a simulation anyway, so we are not claiming real temporal order.
    This keeps the drift demo clean and reliable to run live, and gives the
    training baseline (month 1) a consistent number of fraud examples.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bounds = np.linspace(0, len(df), n_months + 1).astype(int)
    return [df.iloc[bounds[i] : bounds[i + 1]].copy() for i in range(n_months)]


def inject_amount_inflation(batch: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Simulate 'transactions got bigger this month' by scaling `amount` up.
    A PSI check on `amount` will see the shifted distribution."""
    batch = batch.copy()
    batch["amount"] = batch["amount"] * factor
    return batch


def inject_type_shift(
    batch: pd.DataFrame, to_transfer_frac: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Simulate 'more money moved via TRANSFER this month' by relabeling a
    fraction of non-TRANSFER rows to TRANSFER. A PSI check on `type` catches it."""
    batch = batch.copy()
    non_transfer = batch.index[batch["type"] != "TRANSFER"]
    k = int(len(non_transfer) * to_transfer_frac)
    if k > 0:
        chosen = rng.choice(non_transfer.to_numpy(), size=k, replace=False)
        batch.loc[chosen, "type"] = "TRANSFER"
    return batch


def apply_drift(
    batch: pd.DataFrame, plan: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Dispatch to the right drift function based on the plan for this month."""
    kind = plan["kind"]
    if kind == "amount_inflation":
        return inject_amount_inflation(batch, plan["factor"])
    if kind == "type_shift":
        return inject_type_shift(batch, plan["to_transfer_frac"], rng)
    raise ValueError(f"Unknown drift kind: {kind}")


def print_summary(rows: list[dict]) -> None:
    """Print a per-month table so the injected drift is visible at a glance
    (mean_amount jumps on inflation months, %transfer jumps on the type-shift)."""
    header = f"{'month':>5} {'rows':>8} {'frauds':>7} {'fraud_rate':>11} {'mean_amount':>12} {'%transfer':>10}  drift"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['month']:>5} {r['rows']:>8,} {r['frauds']:>7,} "
            f"{r['fraud_rate']:>10.4%} {r['mean_amount']:>12,.0f} "
            f"{r['pct_transfer']:>9.1%}  {r['drift']}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split PaySim into 12 monthly batches with injected drift."
    )
    p.add_argument(
        "--raw", type=Path, default=DEFAULT_RAW, help="Path to raw PaySim CSV."
    )
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output folder for batches."
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=DEFAULT_SUBSAMPLE,
        help="Rows to keep (fraud-rate preserving). 0 = use all rows.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_raw(args.raw)
    print(f"Loaded {len(df):,} rows from {args.raw}")

    df = subsample(df, args.subsample, args.seed)
    print(
        f"Using {len(df):,} rows after subsample (fraud rate {df['isFraud'].mean():.4%})"
    )

    months = split_into_months(df, N_MONTHS, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for month_idx, batch in enumerate(months, start=1):
        drift = DRIFT_PLAN.get(month_idx)
        if drift:
            batch = apply_drift(batch, drift, rng)
        out_path = args.out / f"month_{month_idx:02d}.csv"
        batch.to_csv(out_path, index=False)
        summary.append(
            {
                "month": month_idx,
                "rows": len(batch),
                "frauds": int(batch["isFraud"].sum()),
                "fraud_rate": batch["isFraud"].mean(),
                "mean_amount": batch["amount"].mean(),
                "pct_transfer": (batch["type"] == "TRANSFER").mean(),
                "drift": drift["kind"] if drift else "-",
            }
        )

    print_summary(summary)
    print(
        f"\nWrote {len(months)} monthly batches to {args.out}/  (month_01.csv ... month_12.csv)"
    )


if __name__ == "__main__":
    main()