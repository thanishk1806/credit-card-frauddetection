#!/usr/bin/env python3
"""
Generates a small SYNTHETIC dataset with the same schema as the Kaggle
"creditcard.csv" (Time, V1..V28, Amount, Class) for local development and
testing WITHOUT the real dataset.

This is NOT a substitute for the real Kaggle data for academic evaluation -
use it only to verify that the pipeline runs end-to-end. Replace
data/creditcard.csv with the real Kaggle file before generating results
for your report/viva.

Usage:
    python scripts/generate_sample_data.py --rows 20000 --fraud-rate 0.0017
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(rows * fraud_rate))
    n_legit = rows - n_fraud

    def make_block(n, fraud: bool):
        # Legit transactions: PCA components centered near 0.
        # Fraud transactions: shifted mean + higher variance on a few
        # components, loosely mimicking the separability seen in the real
        # dataset, purely for pipeline-testing purposes.
        v = rng.normal(loc=0.0, scale=1.0, size=(n, 28))
        if fraud:
            shift = np.zeros(28)
            shift[[3, 9, 13, 16]] = [3.0, -3.0, -4.0, -3.0]  # mimic V4, V10, V14, V17
            v = v + shift + rng.normal(0, 1.5, size=(n, 28))
        time = rng.uniform(0, 172792, size=n)
        amount = np.abs(rng.normal(loc=88 if not fraud else 120, scale=250, size=n))
        label = np.ones(n, dtype=int) if fraud else np.zeros(n, dtype=int)
        data = {"Time": time}
        for i in range(28):
            data[f"V{i + 1}"] = v[:, i]
        data["Amount"] = amount
        data["Class"] = label
        return pd.DataFrame(data)

    df = pd.concat([make_block(n_legit, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--fraud-rate", type=float, default=0.0017)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.fraud_rate)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows ({int(df['Class'].sum())} fraud) to {out_path}")


if __name__ == "__main__":
    main()
