"""
Threshold-free comparison of the AAE-aligned and mainstream groups, for both classifiers.
Uses hate_score_1 and hate_score_2 (probability of the hate class) saved by the classification stage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
ID_COLUMN = "Unnamed: 0"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
FLAG_RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
MAINSTREAM_FPRS = [0.05, 0.10, 0.20, 0.30]
POSITIVE_CLASSES = {"1": [0], "2": [0, 1]}

def group_arrays(data):
    aae = data[data["dialect_group"] == "aae_aligned"]
    main = data[data["dialect_group"] == "mainstream"]
    return aae, main

def bootstrap_difference(statistic, aae, main):
    values = []
    for _ in range(N_BOOTSTRAP):
        a = aae.iloc[rng.integers(0, len(aae), len(aae))]
        m = main.iloc[rng.integers(0, len(main), len(main))]
        values.append(statistic(a) - statistic(m))
    return np.array(values)

def interval(values):
    return np.percentile(values, [2.5, 97.5])

rng = np.random.default_rng(RANDOM_SEED)
df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN)
df = df[df["dialect_group"].isin(["aae_aligned", "mainstream"])].reset_index(drop=True)
print(f"{len(df)} tweets: {(df['dialect_group'] == 'aae_aligned').sum()} AAE-aligned, " f"{(df['dialect_group'] == 'mainstream').sum()} mainstream")

for model_name, score_column, model_key in [("Model 1 (twitter-roberta-base-hate)", "hate_score_1", "1"), ("Model 2 (unbiased-toxic-roberta)", "hate_score_2", "2")]:
    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)
    df["true_label"] = df["class"].isin(POSITIVE_CLASSES[model_key]).astype(int)
    aae, main = group_arrays(df)

    # --- 1. AUC per group: how well the score separates hate from not-hate ---
    # AUC = chance that a random hate tweet gets a higher score than a random not-hate tweet.
    # 0.5 = no better than chance, 1 = perfect. It does not depend on any threshold.
    print("\n--- 1. AUC per group (positive vs negative class of this model, Davidson labels) ---")
    auc = lambda data: roc_auc_score(data["true_label"], data[score_column])
    auc_aae, auc_main = auc(aae), auc(main)
    gap_values = bootstrap_difference(auc, aae, main)
    low, high = interval(gap_values)
    print(f"AAE-aligned AUC: {auc_aae:.3f}    mainstream AUC: {auc_main:.3f}    gap: {auc_aae - auc_main:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]")

    # --- 2. Among the not-hate tweets: do the scores differ between groups? ---
    # Chance that a random AAE-aligned not-hate tweet gets a higher score than a random mainstream one.
    # 0.5 = the groups get the same scores (no bias at any threshold). Above 0.5 = AAE-aligned scored higher.
    print("\n--- 2. Score difference between groups in the negtive tweets ---")
    not_hate = df[df["true_label"] == 0]
    aae_nh, main_nh = group_arrays(not_hate)
    def group_auc(a, m):
        labels = np.r_[np.ones(len(a)), np.zeros(len(m))]
        return roc_auc_score(labels, np.r_[a[score_column].to_numpy(), m[score_column].to_numpy()])
    observed = group_auc(aae_nh, main_nh)
    values = []
    for _ in range(N_BOOTSTRAP):
        a = aae_nh.iloc[rng.integers(0, len(aae_nh), len(aae_nh))]
        m = main_nh.iloc[rng.integers(0, len(main_nh), len(main_nh))]
        values.append(group_auc(a, m))
    low, high = interval(np.array(values))
    print(f"P(AAE score > mainstream score) = {observed:.3f}   95% CI [{low:.3f}, {high:.3f}]   (n = {len(aae_nh)} / {len(main_nh)})")
    print(f"Median score: AAE-aligned {aae_nh[score_column].median():.3f}   mainstream {main_nh[score_column].median():.3f}")

    # --- 3. FPR at a fixed overall flag rate (one shared threshold for both groups) ---
    # The threshold is chosen so that the model flags this share of ALL tweets. Same model, same threshold.
    print("\n--- 3. FPR at a fixed overall flag rate (same threshold for both groups) ---")
    print(f"{'flag rate':<12}{'threshold':<12}{'FPR AAE':<10}{'FPR main':<10}{'gap':<10}")
    for rate in FLAG_RATES:
        threshold = np.quantile(df[score_column], 1 - rate)
        fpr_aae = (aae_nh[score_column] >= threshold).mean()
        fpr_main = (main_nh[score_column] >= threshold).mean()
        print(f"{rate:<12.2f}{threshold:<12.4f}{fpr_aae:<10.3f}{fpr_main:<10.3f}{fpr_aae - fpr_main:<+10.3f}")

    # --- 4. FPR of the AAE-aligned group when the mainstream FPR is fixed ---
    # Pick the threshold that gives the mainstream group a chosen FPR, then look at the AAE-aligned FPR.
    # If the model treated both groups alike, the AAE-aligned FPR would be about the same.
    print("\n--- 4. AAE-aligned FPR when the mainstream FPR is fixed ---")
    print(f"{'main FPR':<12}{'threshold':<12}{'achieved':<10}{'FPR AAE':<10}{'ratio':<8}")
    for target in MAINSTREAM_FPRS:
        threshold = np.quantile(main_nh[score_column], 1 - target)
        achieved = (main_nh[score_column] >= threshold).mean()
        fpr_aae = (aae_nh[score_column] >= threshold).mean()
        ratio = fpr_aae / achieved if achieved > 0 else float("nan")
        print(f"{target:<12.2f}{threshold:<12.4f}{achieved:<10.3f}{fpr_aae:<10.3f}{ratio:<8.2f}")