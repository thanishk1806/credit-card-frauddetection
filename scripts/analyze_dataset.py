import pandas as pd
import numpy as np
from pathlib import Path
import json

data_path = Path("data/creditcard.csv")
print(f"Loading {data_path}...")
df = pd.read_csv(data_path)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes.value_counts()}")
print(f"Missing values:\n{df.isnull().sum().sum()}")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

class_counts = df['Class'].value_counts()
print(f"Class distribution:\n{class_counts}")
print(f"Fraud ratio: {class_counts[1] / len(df) * 100:.4f}%")

print("\nAmount summary:")
print(df['Amount'].describe())
print(f"Amount 0 count: {(df['Amount'] == 0).sum()}")
print(f"Amount > 1000 count: {(df['Amount'] > 1000).sum()}")

print("\nTime summary:")
print(df['Time'].describe())

# Correlations with Class
corr = df.corr()['Class'].sort_values()
print("\nTop negative correlations with Class:")
print(corr.head(5))
print("\nTop positive correlations with Class:")
print(corr.tail(6))

# Check time sorting
is_sorted = df['Time'].is_monotonic_increasing
print(f"\nIs Time sorted: {is_sorted}")
