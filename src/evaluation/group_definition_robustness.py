"""
Does the gap depend on how the dialect groups are defined?
Repeats the main comparison (AAE-aligned vs mainstream) with several definitions:
  1. baseline: AAE score >= 0.6 vs <= 0.4 (the one used in the report)
  2. argmax:   the largest component of the 4-way posterior is AAE vs White
  3. masked:   AAE score computed after removing profanity from the text, same 0.6 / 0.4 thresholds
  4. stricter thresholds on the AAE score: 0.7 / 0.3, 0.8 / 0.2, 0.9 / 0.1
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
ID_COLUMN = "Unnamed: 0"
N_BOOTSTRAP = 2000
N_BOOTSTRAP_AUC = 500
MIN_GROUP = 30
RANDOM_SEED = 42
DEFINITIONS = [("1. baseline (0.6 / 0.4)", "group_baseline"),
               ("2. argmax (AAE vs White)", "group_argmax"),
               ("3. profanity removed before scoring", "group_masked"),
               ("4a. stricter thresholds 0.7 / 0.3", "group_0.7"),
               ("4b. stricter thresholds 0.8 / 0.2", "group_0.8"),
               ("4c. stricter thresholds 0.9 / 0.1", "group_0.9")]
HEADER = f"{'definition':<38}{'n AAE':>7}{'n main':>8}{'AAE':>9}{'main':>9}{'gap':>8}   95% CI"

def groups(column):
    return (df[column] == "aae_aligned").to_numpy(), (df[column] == "mainstream").to_numpy()

def gap_with_ci(a, m):
    gaps = [a[rng.integers(0, len(a), len(a))].mean() - m[rng.integers(0, len(m), len(m))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(gaps, [2.5, 97.5])
    return a.mean() - m.mean(), low, high

def flag_row(name, population, model_key, aae, main):
    """Print the flag rate gap for the tweets in `population` (boolean array)."""
    flag = df[f"flag_{model_key}"].to_numpy().astype(float)
    a, m = flag[aae & population], flag[main & population]
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        print(f"{name:<38}{len(a):>7}{len(m):>8}   too few tweets in one group")
        return
    gap, low, high = gap_with_ci(a, m)
    print(f"{name:<38}{len(a):>7}{len(m):>8}{a.mean():>9.3f}{m.mean():>9.3f}{gap:>+8.3f}   [{low:+.3f}, {high:+.3f}]")

def standardised_row(name, model_key, aae, main):
    """Model-1 gap in the not-hate tweets with the mainstream group re-weighted to the label mix (offensive / neither) of the AAE group."""
    flag = df[f"flag_{model_key}"].to_numpy().astype(float)
    pool = CLASS != 0

    def compute(ai, mi):
        total, used = 0.0, 0.0
        for value in (1, 2):
            weight = (CLASS[ai] == value).mean()
            main_in = flag[mi][CLASS[mi] == value]
            if weight == 0 or len(main_in) == 0:
                continue
            total += weight * main_in.mean()
            used += weight
        return np.nan if used == 0 else flag[ai].mean() - total / used

    ai, mi = np.where(aae & pool)[0], np.where(main & pool)[0]
    if len(ai) < MIN_GROUP or len(mi) < MIN_GROUP:
        print(f"{name:<38}{len(ai):>7}{len(mi):>8}   too few tweets in one group")
        return
    boot = [compute(rng.choice(ai, len(ai)), rng.choice(mi, len(mi))) for _ in range(N_BOOTSTRAP)]
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    observed = flag[ai].mean() - flag[mi].mean()
    print(f"{name:<38}{len(ai):>7}{len(mi):>8}   observed {observed:+.3f}   standardised on label {compute(ai, mi):+.3f}   [{low:+.3f}, {high:+.3f}]")

def probability_row(name, population, model_key, aae, main):
    score = df[f"score_{model_key}"].to_numpy()
    a, m = score[aae & population], score[main & population]
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        print(f"{name:<38}{len(a):>7}{len(m):>8}   too few tweets in one group")
        return
    labels = lambda x, y: np.r_[np.ones(len(x)), np.zeros(len(y))]
    value = roc_auc_score(labels(a, m), np.r_[a, m])
    boot = []
    for _ in range(N_BOOTSTRAP_AUC):
        x, y = a[rng.integers(0, len(a), len(a))], m[rng.integers(0, len(m), len(m))]
        boot.append(roc_auc_score(labels(x, y), np.r_[x, y]))
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{name:<38}{len(a):>7}{len(m):>8}{value:>10.3f}   [{low:.3f}, {high:.3f}]")

rng = np.random.default_rng(RANDOM_SEED)

df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN).rename(columns={"pred_label_1": "flag_1", "pred_label_2": "flag_2", "hate_score_1": "score_1", "hate_score_2": "score_2"})
df["true_label"] = (df["class"] == 0).astype(int)  # 1 = hate speech, 0 = not hate
df["group_baseline"] = df["dialect_group"].where(df["dialect_group"].isin(["aae_aligned", "mainstream"]))
df["group_argmax"] = df["argmax_group"].map({"aae_argmax": "aae_aligned", "white_argmax": "mainstream"})
df["group_masked"] = np.select([df["aae_score_masked"] >= 0.6, df["aae_score_masked"] <= 0.4], ["aae_aligned", "mainstream"], default=None)
df.loc[df["aae_score_masked"].isna(), "group_masked"] = None
for upper in [0.7, 0.8, 0.9]:
    df[f"group_{upper}"] = np.select([df["aae_score"] >= upper, df["aae_score"] <= 1 - upper], ["aae_aligned", "mainstream"], default=None)
    df.loc[df["aae_score"].isna(), f"group_{upper}"] = None

CLASS = df["class"].to_numpy()  # 0 hate speech, 1 offensive language, 2 neither

print("--- 1. Label mix of the groups under each definition (% of the group) ---")
print(f"{'definition':<38}{'group':<13}{'n':>7}{'hate':>8}{'offensive':>11}{'neither':>9}")
for name, column in DEFINITIONS:
    for group_name, mask in zip(["AAE-aligned", "mainstream"], groups(column)):
        n = mask.sum()
        shares = [(CLASS[mask] == c).mean() * 100 if n else float("nan") for c in (0, 1, 2)]
        print(f"{name:<38}{group_name:<13}{n:>7}{shares[0]:>8.1f}{shares[1]:>11.1f}{shares[2]:>9.1f}")

print("\n--- 2. Model 1 (hate speech detector): flag rate at the default threshold, by label ---")
# FPR = share flagged as hate among tweets that are not hate speech. Each label separately, because the mix changes with the definition
for population_name, population in [("not-hate (offensive + neither)", CLASS != 0), ("offensive language only", CLASS == 1), ("neither only", CLASS == 2)]:
    print(f"\n{population_name}\n{HEADER}")
    for name, column in DEFINITIONS:
        aae, main = groups(column)
        flag_row(name, population, "1", aae, main)
print(f"\nnot-hate tweets, gap standardised on the label mix of the AAE group\n{'definition':<38}{'n AAE':>7}{'n main':>8}")
for name, column in DEFINITIONS:
    aae, main = groups(column)
    standardised_row(name, "1", aae, main)
print("\n(hate tweets in the AAE group, per definition: " + ", ".join(
    f"{name.split('.')[0]}: {int((groups(column)[0] & (CLASS == 0)).sum())}" for name, column in DEFINITIONS) + ")")

print("\n--- 3. Model 2 (toxicity detector): flag rate at the default threshold within 'neither' ---")
# The 'not-hate' pool is not used, mixes offensive tweets, which are positives for a toxicity model
print(HEADER)
for name, column in DEFINITIONS:
    aae, main = groups(column)
    flag_row(name, CLASS == 2, "2", aae, main)

print("\n--- 4. Without a threshold: P(AAE score > mainstream score) among the tweets labelled 'neither' (0.5 = same scores) ---")
for model_key, model_name in [("1", "model 1"), ("2", "model 2")]:
    print(f"\n{model_name}\n{'definition':<38}{'n AAE':>7}{'n main':>8}{'P(A>M)':>10}   95% CI")
    for name, column in DEFINITIONS:
        aae, main = groups(column)
        probability_row(name, CLASS == 2, model_key, aae, main)

print("\n--- 5. Overlap with the baseline groups ---")
baseline_aae, baseline_main = groups("group_baseline")
print(f"{'definition':<38}{'AAE kept':>9}{'main kept':>10}{'Jaccard AAE':>13}   composition of the new AAE group (baseline AAE / baseline mainstream / other)")
for name, column in DEFINITIONS[1:]:
    new_aae, new_main = groups(column)
    kept_aae = (baseline_aae & new_aae).sum() / baseline_aae.sum()
    kept_main = (baseline_main & new_main).sum() / baseline_main.sum()
    jaccard = (baseline_aae & new_aae).sum() / (baseline_aae | new_aae).sum()
    n_new = max(new_aae.sum(), 1)
    from_aae, from_main = (baseline_aae & new_aae).sum() / n_new, (baseline_main & new_aae).sum() / n_new
    print(f"{name:<38}{kept_aae:>9.1%}{kept_main:>10.1%}{jaccard:>13.3f}   {from_aae:.1%} / {from_main:.1%} / {1 - from_aae - from_main:.1%}   (n = {new_aae.sum()})")

# print("\n--- 6. Manual spot-check of the dialect scores (raw tweets, offensive; mask them in the report) ---")
# scored = df.dropna(subset=["aae_score"])
# def random_sample(data, n):
#     return data.sample(min(n, len(data)), random_state=RANDOM_SEED)


# samples = [("random tweets close to the 0.6 cut-off (0.55 - 0.65)", random_sample(scored[scored["aae_score"].between(0.55, 0.65)], 10)),
#            ("the 5 lowest AAE scores", scored.sort_values("aae_score").head(5)),
#            ("random tweets with a score above 0.9", random_sample(scored[scored["aae_score"] > 0.9], 5))]
# for title, sample in samples:
#     print(f"\n--- {title} ---")
#     for _, row in sample.iterrows():
#         print(f"\nscore={row['aae_score']:.3f} | true={row['label']} | model 1={'hate' if row['flag_1'] == 1 else 'not_hate'}")
#         print(str(row["tweet"])[:140])