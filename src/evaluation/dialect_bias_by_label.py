"""
Compare how often each model flags tweets as hate, per group, within each human label.
"""

import os
import pandas as pd

DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
MODEL_NAMES = {"1": "twitter-roberta-base-hate", "2": "unbiased-toxic-roberta"}
ID_COLUMN = "Unnamed: 0"

data = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN)
data = data[data["dialect_group"].isin(["aae_aligned", "mainstream"])]

for model_key, model_name in MODEL_NAMES.items():
    flag_column = f"pred_label_{model_key}"
    print("\n\n" + "=" * 90)
    print(f"MODEL {model_key}: {model_name}")
    print("=" * 90)
    flag_rate = data.pivot_table(index="label", columns="dialect_group", values=flag_column, aggfunc="mean") * 100
    counts = data.pivot_table(index="label", columns="dialect_group", values=flag_column, aggfunc="count")
    print("--- % of tweets the model flags as hate, by human label ---")
    print(flag_rate.round(1))
    print("\n--- number of tweets ---")
    print(counts)
