"""
Keyword control
is the gap explained by a few women-directed profane words that are common in the AAE-aligned tweets?
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
ID_COLUMN = "Unnamed: 0"
TEXT_COLUMN = "tweet"
N_BOOTSTRAP = 2000
N_BOOTSTRAP_AUC = 500
MIN_GROUP = 30
RANDOM_SEED = 42
# Women-directed profanity, including spelling variants
TERMS_REGEX = re.compile(r"\b(b+[i1!*]+t?ch\w*|bitch\w*|bich\w*|ho+e?[sz]*|hos|pus+y|pus+ies|cunt\w*|slut\w*|wh[o0]re\w*|skank\w*)\b", re.IGNORECASE)
MODELS = [("1", "model 1"), ("2", "model 2")]

def gap_row(label, data, model_key):
    flag, aae = data[f"flag_{model_key}"].to_numpy().astype(float), data["is_aae"].to_numpy()
    a, m = flag[aae], flag[~aae]
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        print(f"{label:<34}model {model_key}{len(a):>6}{len(m):>7}   too few tweets in one group")
        return
    gaps = [a[rng.integers(0, len(a), len(a))].mean() - m[rng.integers(0, len(m), len(m))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(gaps, [2.5, 97.5])
    print(f"{label:<34}model {model_key}{len(a):>6}{len(m):>7}{a.mean():>8.3f}{m.mean():>8.3f}{a.mean() - m.mean():>+8.3f}   [{low:+.3f}, {high:+.3f}]")

def probability_row(label, data, model_key):
    score, aae = data[f"score_{model_key}"].to_numpy(), data["is_aae"].to_numpy()
    a, m = score[aae], score[~aae]
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        print(f"{label:<34}model {model_key}{len(a):>6}{len(m):>7}   too few tweets in one group")
        return
    labels = lambda x, y: np.r_[np.ones(len(x)), np.zeros(len(y))]
    value = roc_auc_score(labels(a, m), np.r_[a, m])
    boot = []
    for _ in range(N_BOOTSTRAP_AUC):
        x, y = a[rng.integers(0, len(a), len(a))], m[rng.integers(0, len(m), len(m))]
        boot.append(roc_auc_score(labels(x, y), np.r_[x, y]))
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{label:<34}model {model_key}{len(a):>6}{len(m):>7}{np.median(a):>9.3f}{np.median(m):>9.3f}{value:>9.3f}   [{low:.3f}, {high:.3f}]")

def effect_of_terms(label, data, model_key):
    """Flag rate with a term minus without a term, inside each group."""
    flag, aae, term = data[f"flag_{model_key}"].to_numpy().astype(float), data["is_aae"].to_numpy(), data["has_term"].to_numpy()
    out = f"{label:<34}model {model_key}"
    for name, group in [("AAE-aligned", aae), ("mainstream", ~aae)]:
        w, wo = flag[group & term], flag[group & ~term]
        if len(w) < MIN_GROUP or len(wo) < MIN_GROUP:
            out += f"   {name}: too few tweets ({len(w)} with / {len(wo)} without)"
            continue
        boot = [w[rng.integers(0, len(w), len(w))].mean() - wo[rng.integers(0, len(wo), len(wo))].mean() for _ in range(N_BOOTSTRAP)]
        low, high = np.percentile(boot, [2.5, 97.5])
        out += f"   {name}: {w.mean():.3f} with / {wo.mean():.3f} without = {w.mean() - wo.mean():+.3f} [{low:+.3f}, {high:+.3f}]"
    print(out)

def adjusted_gap(label, data, model_key):
    """AAE rate minus the mainstream rate re-weighted to the AAE share of tweets with a term."""
    flag, aae, term = data[f"flag_{model_key}"].to_numpy().astype(float), data["is_aae"].to_numpy(), data["has_term"].to_numpy()

    def compute(ai, mi):
        weights, total, used = [], 0.0, 0.0
        for value in (True, False):
            weight = (term[ai] == value).mean()
            main_in = flag[mi][term[mi] == value]
            if weight == 0 or len(main_in) == 0:
                continue
            total += weight * main_in.mean()
            used += weight
        return np.nan if used == 0 else flag[ai].mean() - total / used

    ai, mi = np.where(aae)[0], np.where(~aae)[0]
    if len(ai) < MIN_GROUP or len(mi) < MIN_GROUP:
        print(f"{label:<34}model {model_key}   too few tweets in one group")
        return
    boot = [compute(rng.choice(ai, len(ai)), rng.choice(mi, len(mi))) for _ in range(N_BOOTSTRAP)]
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    observed = flag[ai].mean() - flag[mi].mean()
    print(f"{label:<34}model {model_key}   observed {observed:+.3f}   adjusted for the terms {compute(ai, mi):+.3f}   [{low:+.3f}, {high:+.3f}]")

rng = np.random.default_rng(RANDOM_SEED)
df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN).rename(columns={"pred_label_1": "flag_1", "pred_label_2": "flag_2", "hate_score_1": "score_1", "hate_score_2": "score_2"})
df["true_label"] = (df["class"] == 0).astype(int)  # 1 = hate speech, 0 = not hate
df = df[df["dialect_group"].isin(["aae_aligned", "mainstream"])].reset_index(drop=True)
df["is_aae"] = df["dialect_group"] == "aae_aligned"
text_without_mentions = df[TEXT_COLUMN].astype(str).str.replace(r"@\w+|http\S+", " ", regex=True)
df["has_term"] = text_without_mentions.apply(lambda text: bool(TERMS_REGEX.search(text)))

offensive = df[df["class"] == 1]
not_hate = df[df["true_label"] == 0]
neither = df[df["class"] == 2]
SUBSETS = [("offensive, WITH a term", offensive[offensive["has_term"]]), ("offensive, WITHOUT any term", offensive[~offensive["has_term"]]),
           ("offensive, all", offensive), ("not-hate, WITH a term", not_hate[not_hate["has_term"]]),
           ("not-hate, WITHOUT any term", not_hate[~not_hate["has_term"]]), ("not-hate, all", not_hate),
           ("neither only", neither)]

print("\n--- 1. How common are the terms? ---")
for name, data in [("offensive language", offensive), ("hate speech", df[df["class"] == 0]), ("neither", neither)]:
    shares = data.groupby("dialect_group")["has_term"].mean()
    print(f"{name:<20} AAE-aligned {shares.get('aae_aligned', float('nan')):.3f}   mainstream {shares.get('mainstream', float('nan')):.3f}")

print("\n--- 2. Flag rate gap at the default threshold, split by the terms ---")
# Gap = AAE-aligned minus mainstream, CI by bootstrap.
# Model 1: share flagged as hate. Model 2: share flagged as toxic.
# For model 2 the rows on offensive-language tweets measure RECALL of the toxicity detector, not false positives
print(f"{'subset':<34}{'model':<8}{'n AAE':>6}{'n main':>7}{'AAE':>8}{'main':>8}{'gap':>8}   95% CI")
for label, data in SUBSETS:
    for key, _ in MODELS:
        if key == "2" and label.startswith("not-hate"):
            continue  # for model 2 'not-hate' mixes offensive tweets (positives) and neither (negatives): read 'neither only'
        gap_row(label, data, key)

print("\n--- 3. Same splits without a threshold P(AAE score > mainstream score) ---")
# 0.5 = the groups get the same scores. Median = median score of each group.
print(f"{'subset':<34}{'model':<8}{'n AAE':>6}{'n main':>7}{'med AAE':>9}{'med main':>9}{'P(A>M)':>9}   95% CI")
for label, data in SUBSETS:
    for key, _ in MODELS:
        if key == "2" and label.startswith("not-hate"):
            continue
        probability_row(label, data, key)

print("\n--- 4. How much does each group react to the terms? ---")
for label, data in [("offensive language", offensive), ("neither", neither)]:
    for key, _ in MODELS:
        effect_of_terms(label, data, key)

print("\n--- 5. Gap after giving the mainstream group the same share of tweets with a term as the AAE-aligned group ---")
for label, data in [("not-hate (model 1) ", not_hate), ("neither", neither)]:
    for key, _ in MODELS:
        if key == "2" and label.startswith("not-hate"):
            continue
        adjusted_gap(label, data, key)