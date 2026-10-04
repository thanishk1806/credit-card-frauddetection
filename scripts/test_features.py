import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("data/creditcard.csv")
print("Original shape:", df.shape)

# Sort by Time to ensure chronological order
df = df.sort_values("Time").reset_index(drop=True)

# 1. Time features
df["transaction_hour"] = (df["Time"] / 3600.0) % 24.0
df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24.0)

# 2. Amount features
df["log_amount"] = np.log1p(df["Amount"])

# 3. Sequential / Temporal features (strictly backward looking)
time_diff = df["Time"].diff().fillna(0.0)
df["time_since_prev"] = time_diff.clip(lower=0.0)

# Rolling velocity (number of transactions in the last 300 seconds / 5 minutes)
# We can compute rolling counts using searchsorted on Time
times = df["Time"].values
# For each i, find how many transactions occurred in [Time[i] - 300, Time[i]]
idx_300 = np.searchsorted(times, times - 300.0, side='left')
df["tx_velocity_5m"] = np.arange(len(df)) - idx_300 + 1

print("Engineered features preview:")
print(df[["Time", "transaction_hour", "hour_sin", "hour_cos", "Amount", "log_amount", "time_since_prev", "tx_velocity_5m"]].head())

corr = df[["transaction_hour", "hour_sin", "hour_cos", "Amount", "log_amount", "time_since_prev", "tx_velocity_5m", "Class"]].corr()["Class"]
print("\nCorrelations with Class:")
print(corr)
