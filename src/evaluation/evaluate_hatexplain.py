"""
Evaluate both classifiers on HateXplain
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIALECT_PATH = os.path.join("data", "processed", "hatexplain_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "hatexplain_predictions.csv")
ID_COLUMN = "post_id"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
MAINSTREAM_FPRS = [0.10, 0.20, 0.30]
rng = np.random.default_rng(RANDOM_SEED)

df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN)
df = df[df["dialect_group"].isin(["aae_aligned", "mainstream"])].reset_index(drop=True)
df["true_label"] = (df["class"] == 0).astype(int)  # 1 = hate speech, 0 = not hate
# Model 1 detects hate (positive = hate speech). Model 2 detects toxicity, so for it positive = hate speech OR
# offensive language and its real false positives are the posts labelled normal.
POSITIVE_CLASSES = {"1": [0], "2": [0, 1]}
LABEL_NAMES = {0: "hate speech", 1: "offensive language", 2: "neither"}

print("--- Corpus: groups and human labels ---")
table = pd.crosstab(df["dialect_group"], df["class"].map(LABEL_NAMES))
table["total"] = table.sum(axis=1)
print(table.to_string())
print("\n",(pd.crosstab(df["dialect_group"], df["class"].map(LABEL_NAMES), normalize="index") * 100).round(1).to_string())

def gap_with_ci(aae_values, main_values):
    observed = aae_values.mean() - main_values.mean()
    gaps = [aae_values[rng.integers(0, len(aae_values), len(aae_values))].mean()
            - main_values[rng.integers(0, len(main_values), len(main_values))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(gaps, [2.5, 97.5])
    return observed, low, high

for key, model_name in [("1", "Model 1 (twitter-roberta-base-hate)"), ("2", "Model 2 (unbiased-toxic-roberta)")]:
    flag = df[f"pred_label_{key}"].to_numpy()
    score = df[f"hate_score_{key}"].to_numpy()
    is_aae = (df["dialect_group"] == "aae_aligned").to_numpy()
    true = df["true_label"].to_numpy()
    negative = ~df["class"].isin(POSITIVE_CLASSES[key]).to_numpy()  # real negatives of this model
    classes = df["class"].to_numpy()
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    print("\n--- 1. Flag rates at the default threshold ---")
    if key == "2":
        print("(Model 2 detects toxicity: only the row 'FPR within neither' is a real false positive rate; the other rows are flag rates.)")
    print(f"{'subset':<30}{'n AAE':>7}{'n main':>8}{'AAE':>8}{'main':>8}{'gap':>8}   95% CI")
    subsets = [("FPR (all not-hate)", true == 0), ("FPR within 'neither'", classes == 2),
               ("FPR within 'offensive language'", classes == 1), ("Recall (hate speech)", true == 1)]
    for name, mask in subsets:
        a, m = flag[mask & is_aae], flag[mask & ~is_aae]
        if len(a) < 30 or len(m) < 30:
            print(f"{name:<30}{len(a):>7}{len(m):>8}   too few posts in one group")
            continue
        gap, low, high = gap_with_ci(a, m)
        print(f"{name:<30}{len(a):>7}{len(m):>8}{a.mean():>8.3f}{m.mean():>8.3f}{gap:>+8.3f}   [{low:+.3f}, {high:+.3f}]")

    print("\n--- 2. Threshold-free, among the real negatives (not hate for model 1, normal for model 2) ---")
    nh = negative
    a_scores, m_scores = score[nh & is_aae], score[nh & ~is_aae]
    labels = np.r_[np.ones(len(a_scores)), np.zeros(len(m_scores))]
    observed = roc_auc_score(labels, np.r_[a_scores, m_scores])
    values = []
    for _ in range(N_BOOTSTRAP):
        a = a_scores[rng.integers(0, len(a_scores), len(a_scores))]
        m = m_scores[rng.integers(0, len(m_scores), len(m_scores))]
        values.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(m))], np.r_[a, m]))
    low, high = np.percentile(values, [2.5, 97.5])
    print(f"P(AAE score > mainstream score) = {observed:.3f}   95% CI [{low:.3f}, {high:.3f}]   (0.5 = equal scores)")
    print(f"{'main FPR fixed at':<20}{'achieved':>10}{'AAE FPR':>10}{'ratio':>8}")
    for target in MAINSTREAM_FPRS:
        threshold = np.quantile(m_scores, 1 - target)
        achieved = (m_scores >= threshold).mean()
        fpr_aae = (a_scores >= threshold).mean()
        print(f"{target:<20.2f}{achieved:>10.3f}{fpr_aae:>10.3f}{fpr_aae / achieved:>8.2f}")