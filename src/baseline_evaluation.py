"""
Evaluate the roberta predictions: compare the aae_aligned and mainstream groups
on precision, recall, F1, accuracy and FPR (false positive rate), and test with
a bootstrap whether the F1 and FPR gaps between the groups are significant.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

IN_PATH = os.path.join("data", "processed", "davidson_dataset_with_dialect_and_roberta_classification.csv")
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def confusion_counts(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return tp, fp, fn, tn


def divide(a, b):
    return a / b if b > 0 else 0.0


def precision(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(tp, tp + fp)


def recall(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(tp, tp + fn)


def f1(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(2 * tp, 2 * tp + fp + fn)


def accuracy(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(tp + tn, tp + fp + fn + tn)


def false_positive_rate(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(fp, fp + tn)


METRICS = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "false positive rate": false_positive_rate,
}


def bootstrap_gap(metric):
    """Resample each group with replacement and recompute the gap N_BOOTSTRAP times."""
    gaps = []
    for _ in range(N_BOOTSTRAP):
        values = []
        for y_true, y_pred in (groups["aae_aligned"], groups["mainstream"]):
            idx = rng.integers(0, len(y_true), len(y_true))
            values.append(metric(y_true[idx], y_pred[idx]))
        gaps.append(values[0] - values[1])
    return np.array(gaps)


df = pd.read_csv(IN_PATH)
groups = {}
for name in ["aae_aligned", "mainstream"]:
    sub = df[df["dialect_group"] == name]
    groups[name] = (sub["true_label"].to_numpy(), sub["pred_label"].to_numpy())

# --- metrics per group ---
for name, (y_true, y_pred) in groups.items():
    print(f"\n--- {name} (n={len(y_true)}) ---")
    print(classification_report(y_true, y_pred, target_names=["not_hate", "hate"], zero_division=0))

print("\n--- Summary ---")
print(f"{'Metric':<22}{'AAE-aligned':<15}{'Mainstream':<15}{'Gap':<10}")
for metric_name, metric in METRICS.items():
    aae = metric(*groups["aae_aligned"])
    main = metric(*groups["mainstream"])
    print(f"{metric_name:<22}{aae:<15.3f}{main:<15.3f}{aae - main:<+10.3f}")

# --- Bootstrap significance test on the gap between groups ---
for metric_name in ["f1", "false positive rate"]:
    metric = METRICS[metric_name]
    observed_gap = metric(*groups["aae_aligned"]) - metric(*groups["mainstream"])
    boot_gaps = bootstrap_gap(metric) # calculate the gap "creating" N_BOOTSTRAP datasets
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5]) # remove the extremes (cut the 2.5% lowest anf 97.5% highest)
    p_value = 2 * min((boot_gaps >= 0).mean(), (boot_gaps < 0).mean())

    print(f"\n--- Bootstrap ({N_BOOTSTRAP} resamples): {metric_name} gap (AAE - mainstream) ---")
    print(f"Observed gap: {observed_gap:+.3f}")
    print(f"95% bootstrap CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Approx. two-sided p-value: {p_value:.4f}")
    if ci_low > 0 or ci_high < 0:
        print("=> The CI does not include zero: likely a real difference.")
    else:
        print("=> The CI includes zero: not conclusively different.")