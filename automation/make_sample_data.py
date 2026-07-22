"""
Generate a small SYNTHETIC dataset.

Why this exists: the real PaySim file is ~470 MB and is not committed, so CI
runners (and anyone who has not downloaded it) have no data. This script
fabricates a small stand-in with the same columns, so the whole pipeline -
batches, training, drift check, retraining - can run end to end without the
download. It is NOT real data; it exists only to exercise the machinery (for
example inside the scheduled GitHub Action).

Run:
    python automation/make_sample_data.py --rows 120000 --out data/raw/paysim.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_PROBS = [0.34, 0.35, 0.22, 0.08, 0.01]


def make_frame(rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(TYPES, size=rows, p=TYPE_PROBS)
    amount = np.round(rng.lognormal(8.0, 1.2, rows), 2)
    old_org = np.round(rng.lognormal(9.0, 1.3, rows), 2)
    new_org = np.clip(old_org - amount, 0, None)
    old_dest = np.round(rng.lognormal(9.0, 1.3, rows), 2)
    new_dest = old_dest + amount

    # Fraud only happens in TRANSFER / CASH_OUT, like the real PaySim data.
    is_fraud = np.zeros(rows, dtype=int)
    eligible = np.where((types == "TRANSFER") | (types == "CASH_OUT"))[0]
    n_fraud = min(max(1, int(fraud_rate * rows)), len(eligible))
    fraud_idx = rng.choice(eligible, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    # Give fraud a LEARNABLE signature, mirroring real PaySim: the transaction
    # empties the origin account (amount == old balance, new balance 0) into a
    # destination whose balance stays 0. Without a pattern like this there is
    # nothing for the model to learn on synthetic data.
    amount[fraud_idx] = old_org[fraud_idx]
    new_org[fraud_idx] = 0.0
    old_dest[fraud_idx] = 0.0
    new_dest[fraud_idx] = 0.0

    return pd.DataFrame(
        {
            "step": rng.integers(1, 745, rows),
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(rows)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": [f"M{i}" for i in range(rows)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": ((types == "TRANSFER") & (amount > 200000)).astype(int),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic PaySim-shaped data.")
    p.add_argument("--rows", type=int, default=120_000)
    p.add_argument(
        "--fraud-rate",
        type=float,
        default=0.01,
        help="Deliberately higher than real PaySim (~0.0013) so the small "
        "CI batches always contain enough fraud examples to train.",
    )
    p.add_argument("--out", type=Path, default=Path("data/raw/paysim.csv"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = make_frame(args.rows, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    print(
        f"Wrote {len(df):,} synthetic rows ({int(df['isFraud'].sum()):,} fraud) to {args.out}"
    )


if __name__ == "__main__":
    main()
