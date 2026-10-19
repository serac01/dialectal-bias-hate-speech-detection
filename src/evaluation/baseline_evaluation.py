"""
Evaluation for both classifiers on the Davidson data. Compare the aae_aligned and mainstream groups
on precision, recall, F1, accuracy, FPR and test with a bootstrap whether gaps between the groups are significant.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
ID_COLUMN = "Unnamed: 0"
MODEL_NAMES = {"1": "twitter-roberta-base-hate", "2": "roberta-hate-speech-dynabench-r4-target"}
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

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

def share_of_hate(y_true, y_pred):
    """Prevalence: share of tweets that are really hate (Davidson label)."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(tp + fn, tp + fp + fn + tn)


def share_flagged(y_true, y_pred):
    """Share of ALL tweets that the model flagged as hate."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return divide(tp + fp, tp + fp + fn + tn)


def precision_lift(y_true, y_pred):
    """Precision divided by the share of hate. 1 = same as flagging at random."""
    return divide(precision(y_true, y_pred), share_of_hate(y_true, y_pred))


def accuracy_if_never_flagging(y_true, y_pred):
    """Accuracy of a model that never flags anything: 1 - share of hate."""
    return 1 - share_of_hate(y_true, y_pred)


def miss_rate(y_true, y_pred):
    """False negative rate: share of the real hate that the model missed."""
    return 1 - recall(y_true, y_pred)


METRICS = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "false positive rate": false_positive_rate,
}

EXTRA_METRICS = {
    "share of hate (real)": share_of_hate,
    "share flagged (model)": share_flagged,
    "precision lift": precision_lift,
    "accuracy if never flag": accuracy_if_never_flagging,
    "miss rate (FNR)": miss_rate,
}

def bootstrap_gap(metric):  # uses the current `groups` and `rng` of the loop below
    """Resample each group with replacement and recompute the gap N_BOOTSTRAP times."""
    gaps = []
    for _ in range(N_BOOTSTRAP):
        values = []
        for y_true, y_pred in (groups["aae_aligned"], groups["mainstream"]):
            idx = rng.integers(0, len(y_true), len(y_true))
            values.append(metric(y_true[idx], y_pred[idx]))
        gaps.append(values[0] - values[1])
    return np.array(gaps)


data = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN)
data["true_label"] = (data["class"] == 0).astype(int)  # Davidson ground truth: 1 = hate speech, 0 = not hate

for model_key, model_name in MODEL_NAMES.items():
    print("\n\n" + "=" * 90)
    print(f"MODEL {model_key}: {model_name}")
    print("=" * 90)
    rng = np.random.default_rng(RANDOM_SEED)  # same seed for each model
    groups = {}
    for name in ["aae_aligned", "mainstream"]:
        sub = data[data["dialect_group"] == name]
        groups[name] = (sub["true_label"].to_numpy(), sub[f"pred_label_{model_key}"].to_numpy())

    print("\n","-"*30," 1. Confusion matrix per group ","-"*30)
    for name, (y_true, y_pred) in groups.items():
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        print(f"\n{name} (n={len(y_true)})")
        print(f"{'':<20}{'not flagged':<16}{'flagged':<16}")
        print(f"{'real not-hate':<20}{'TN = ' + str(tn):<16}{'FP = ' + str(fp):<16}")
        print(f"{'real hate':<20}{'FN = ' + str(fn):<16}{'TP = ' + str(tp):<16}")

    print("\n","-"*30," 2. Metrics per group","-"*30)
    for name, (y_true, y_pred) in groups.items():
        print(f"\n--- {name} (n={len(y_true)}) ---")
        print(classification_report(y_true, y_pred, target_names=["not_hate", "hate"], zero_division=0))

    print("\n","-"*30," 4. Summary: AAE-aligned vs mainstream ","-"*30)
    print(f"{'Metric':<24}{'AAE-aligned':<14}{'Mainstream':<14}{'Gap':<10}")
    for metric_name, metric in METRICS.items():
        aae = metric(*groups["aae_aligned"])
        main = metric(*groups["mainstream"])
        print(f"{metric_name:<24}{aae:<14.3f}{main:<14.3f}{aae - main:<+10.3f}")

    print("\n","-"*30," 5. Reference numbers ","-"*30)
    print(f"{'Number':<24}{'AAE-aligned':<14}{'Mainstream':<14}")
    for extra_name, extra_metric in EXTRA_METRICS.items():
        aae = extra_metric(*groups["aae_aligned"])
        main = extra_metric(*groups["mainstream"])
        print(f"{extra_name:<24}{aae:<14.3f}{main:<14.3f}")

    print("\n","-"*30," 6. Bootstrap ","-"*30)
    for metric_name in ["f1", "false positive rate"]:
        metric = METRICS[metric_name]
        observed_gap = metric(*groups["aae_aligned"]) - metric(*groups["mainstream"])
        boot_gaps = bootstrap_gap(metric) # calculate the gap "creating" N_BOOTSTRAP datasets
        ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5]) # cut the 2.5% lowest and the 2.5% highest
        p_value = 2 * min((boot_gaps >= 0).mean(), (boot_gaps < 0).mean())
        noise = (ci_high - ci_low) / 2
        signal_to_noise = abs(observed_gap) / noise if noise > 0 else float("inf")
        p_text = f"{p_value:.4f}" if p_value > 0 else f"< {2 / N_BOOTSTRAP:.3f}"

        print(f"\n--- Bootstrap ({N_BOOTSTRAP} resamples): {metric_name} gap (AAE - mainstream) ---")
        print(f"Observed gap: {observed_gap:+.3f}")
        print(f"95% bootstrap CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
        print(f"Noise (half-width of the CI): +-{noise:.3f}")
        print(f"Gap / noise: {signal_to_noise:.1f}")
        print(f"Approx. two-sided p-value: {p_text}")
        if ci_low > 0 or ci_high < 0:
            print("=> The CI does not include zero: likely a real difference.")
        else:
            print("=> The CI includes zero: not conclusively different.")