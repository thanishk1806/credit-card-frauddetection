#!/usr/bin/env python3
"""
Generates a realistic transaction dataset with both Kaggle PCA compatibility
(Time, V1..V28, Amount, Class) AND user-facing banking transaction attributes
(transaction_type, merchant_category, device_type, card_present,
international_transaction, time_since_prev, tx_velocity_5m).

Usage:
    python scripts/generate_sample_data.py --rows 20000 --fraud-rate 0.02
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(rows * fraud_rate))
    n_legit = rows - n_fraud

    # --- Legitimate Transactions ---
    legit_types = ["pos", "mobile_app", "online", "contactless", "atm"]
    legit_type_p = [0.42, 0.28, 0.18, 0.10, 0.02]

    legit_cats = ["grocery", "food_dining", "retail_shopping", "fuel", "utilities", "electronics", "luxury_jewelry"]
    legit_cat_p = [0.30, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02]

    legit_devices = ["mobile", "pos_terminal", "desktop", "atm_kiosk"]
    legit_dev_p = [0.50, 0.35, 0.12, 0.03]

    legit_card_present = rng.choice([True, False], size=n_legit, p=[0.80, 0.20])
    legit_intl = rng.choice([True, False], size=n_legit, p=[0.04, 0.96])
    legit_tx_types = rng.choice(legit_types, size=n_legit, p=legit_type_p)
    legit_merch_cats = rng.choice(legit_cats, size=n_legit, p=legit_cat_p)
    legit_dev_types = rng.choice(legit_devices, size=n_legit, p=legit_dev_p)

    # Legitimate amounts: log-normal centered around $45 (typical consumer spending)
    legit_amounts = np.clip(np.exp(rng.normal(3.8, 0.9, size=n_legit)), 2.0, 1500.0)

    # Legitimate time: normal distribution across days with daytime bias
    legit_hours = (rng.normal(14.0, 4.0, size=n_legit)) % 24.0
    legit_days = rng.integers(0, 2, size=n_legit)
    legit_times = legit_days * 86400.0 + legit_hours * 3600.0 + rng.uniform(0, 3600, size=n_legit)

    legit_time_prev = np.clip(rng.exponential(scale=1800.0, size=n_legit), 60.0, 86400.0)
    legit_velocity = rng.choice([1.0, 2.0, 3.0], size=n_legit, p=[0.94, 0.05, 0.01])

    # Legitimate PCA features: standard normal centered at 0
    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # --- Fraudulent Transactions ---
    fraud_types = ["online", "wire_transfer", "atm", "mobile_app"]
    fraud_type_p = [0.65, 0.20, 0.10, 0.05]

    fraud_cats = ["electronics", "luxury_jewelry", "travel_airlines", "entertainment", "grocery"]
    fraud_cat_p = [0.40, 0.30, 0.15, 0.10, 0.05]

    fraud_devices = ["unknown_device", "desktop", "mobile"]
    fraud_dev_p = [0.55, 0.30, 0.15]

    fraud_card_present = rng.choice([True, False], size=n_fraud, p=[0.05, 0.95])
    fraud_intl = rng.choice([True, False], size=n_fraud, p=[0.60, 0.40])
    fraud_tx_types = rng.choice(fraud_types, size=n_fraud, p=fraud_type_p)
    fraud_merch_cats = rng.choice(fraud_cats, size=n_fraud, p=fraud_cat_p)
    fraud_dev_types = rng.choice(fraud_devices, size=n_fraud, p=fraud_dev_p)

    # Fraud amounts: bimodal - 30% micro-testing ($1 - $15), 70% large ticket ($500 - $12,000)
    is_high_fraud = rng.choice([True, False], size=n_fraud, p=[0.70, 0.30])
    fraud_amounts = np.where(
        is_high_fraud,
        np.exp(rng.normal(7.5, 0.9, size=n_fraud)),  # $800 - $6,000+
        rng.uniform(1.0, 15.0, size=n_fraud),         # $1 - $15
    )
    fraud_amounts = np.clip(fraud_amounts, 1.0, 50000.0)

    # Fraud time: 55% late-night (23:00 - 05:00)
    is_night_fraud = rng.choice([True, False], size=n_fraud, p=[0.55, 0.45])
    night_hours = rng.uniform(0.0, 5.0, size=n_fraud)
    day_hours = rng.uniform(6.0, 23.0, size=n_fraud)
    fraud_hours = np.where(is_night_fraud, night_hours, day_hours)
    fraud_days = rng.integers(0, 2, size=n_fraud)
    fraud_times = fraud_days * 86400.0 + fraud_hours * 3600.0 + rng.uniform(0, 3600, size=n_fraud)

    # Rapid bursts
    fraud_time_prev = np.clip(rng.exponential(scale=25.0, size=n_fraud), 2.0, 300.0)
    fraud_velocity = rng.integers(2, 8, size=n_fraud).astype(float)

    # Fraud PCA shifts
    fraud_v = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    shift = np.zeros(28)
    shift[[3, 9, 10, 11, 13, 15, 16]] = [3.5, -3.5, 2.5, -3.0, -4.5, -2.5, -3.5]
    fraud_v = fraud_v + shift

    # Assemble DataFrames
    legit_df = pd.DataFrame({
        "Time": legit_times,
        "Amount": legit_amounts,
        "transaction_type": legit_tx_types,
        "merchant_category": legit_merch_cats,
        "device_type": legit_dev_types,
        "card_present": legit_card_present,
        "international_transaction": legit_intl,
        "time_since_prev": legit_time_prev,
        "tx_velocity_5m": legit_velocity,
        "Class": np.zeros(n_legit, dtype=int),
    })
    for i in range(28):
        legit_df[f"V{i + 1}"] = legit_v[:, i]

    fraud_df = pd.DataFrame({
        "Time": fraud_times,
        "Amount": fraud_amounts,
        "transaction_type": fraud_tx_types,
        "merchant_category": fraud_merch_cats,
        "device_type": fraud_dev_types,
        "card_present": fraud_card_present,
        "international_transaction": fraud_intl,
        "time_since_prev": fraud_time_prev,
        "tx_velocity_5m": fraud_velocity,
        "Class": np.ones(n_fraud, dtype=int),
    })
    for i in range(28):
        fraud_df[f"V{i + 1}"] = fraud_v[:, i]

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.fraud_rate)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} realistic rows ({int(df['Class'].sum())} fraud, {len(df) - int(df['Class'].sum())} legit) to {out_path}")


if __name__ == "__main__":
    main()
