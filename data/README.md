# Dataset

Place the Kaggle **European Credit Card Fraud Detection** dataset file here as:

```
data/creditcard.csv
```

Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The file must contain the columns: `Time, V1, V2, ..., V28, Amount, Class`.

## No real dataset available yet?

For local development/testing only, you can generate a small synthetic
dataset with the same schema:

```
python scripts/generate_sample_data.py --rows 20000 --fraud-rate 0.0017
```

This lets you verify the entire pipeline (training, dashboard, prediction,
SHAP, PDF reports) runs end-to-end. **Do not use synthetic data for your
actual academic report/viva results** — replace this file with the real
Kaggle `creditcard.csv` before generating final metrics.
