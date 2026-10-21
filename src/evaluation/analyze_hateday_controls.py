"""
Controls for HateDay is the gap between AAE-aligned and mainstream tweets labelled 'neither'
explained by profanity, by tweet length or by which words appear?
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

DIALECT_PATH = os.path.join("data", "processed", "hateday_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "hateday_predictions.csv")
ID_COLUMN = "tweet_id"
TEXT_COLUMN = "tweet"
MIN_GROUP = 30
N_BOOTSTRAP = 2000
N_BOOTSTRAP_AUC = 500
N_BOOTSTRAP_REFIT = 100   # ridge version: the model is refitted in every resample
N_LENGTH_GROUPS = 3
LENGTH_BINS = [(1, 3), (4, 6), (7, 10), (11, 1000)]
MAX_WORDS = 15            # tweets longer than this are put in one stratum (exact-length matching)
N_FOLDS = 5
C_VALUES = [0.1, 1.0, 10.0, 100.0]
ALPHA_VALUES = [0.3, 1.0, 3.0, 10.0]
MIN_DOCUMENTS_LIST = 20
TOP_WORDS = 20
RANDOM_SEED = 42
MODELS = [("1", "Model 1 (twitter-roberta-base-hate)"), ("2", "Model 2 (unbiased-toxic-roberta)")]
rng = np.random.default_rng(RANDOM_SEED)

# Profanity and slurs, with common spelling variants (whole words, case-insensitive). Applied to the cleaned text (no mentions, no links).
PROFANITY = re.compile(
    r"\b(b[i1!*]+t?c?h\w*|biatch\w*|hoe?s?|hoes?z|hos|ho+|pussy|pussies|cunt\w*|slut\w*|whor\w*|skank\w*|thot\w*|"
    r"n[i1!*]+gg\w*|n[i1]cc\w*|nigg\w*|nicca\w*|f+u+c+k\w*|fck\w*|shit\w*|sh[i1]t\w*|ass|asses|azz|dick\w*|d[i1]ck\w*|"
    r"fag\w*|retard\w*|damn\w*|bastard\w*|piss\w*|crap)\b", re.IGNORECASE)

# Tokenizers: "basic" is the earlier setting; "extended" keeps one-letter tokens (u, n, b), numbers and emojis.
CONFIGS = {"basic": {"token_pattern": r"[a-z][a-z']+", "min_df": 20, "emoji": False},
           "extended": {"token_pattern": r"[a-z0-9_']+", "min_df": 10, "emoji": True}}
MAIN_CONFIG = "extended"

def make_cleaner(keep_emoji):
    def clean(text):
        text = str(text).lower()
        if keep_emoji:
            text = re.sub(r"&#(\d+);", r" emoji\1 ", text)
        text = re.sub(r"&#?\w+;", " ", text)
        text = re.sub(r"http\S*", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"\brt\b", " ", text)
        return text
    return clean

clean_basic = make_cleaner(False)

def build_matrix(texts, config):
    vectorizer = CountVectorizer(preprocessor=make_cleaner(config["emoji"]), token_pattern=config["token_pattern"], binary=True, min_df=config["min_df"])
    return vectorizer.fit_transform(texts), np.array(vectorizer.get_feature_names_out())

df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN).rename(columns={"pred_label_1": "flag_1", "pred_label_2": "flag_2", "hate_score_1": "score_1", "hate_score_2": "score_2"})
df = df[df["dialect_group"].isin(["aae_aligned", "mainstream"]) & (df["class"] == 2)].reset_index(drop=True)
is_aae = (df["dialect_group"] == "aae_aligned").to_numpy()
aae_idx, main_idx = np.where(is_aae)[0], np.where(~is_aae)[0]
cleaned = df[TEXT_COLUMN].astype(str).apply(clean_basic)
df["has_profanity"] = cleaned.apply(lambda text: bool(PROFANITY.search(text)))
df["tokens"] = cleaned.apply(lambda text: re.findall(r"[a-z0-9_']+", text))
df["n_words"] = df["tokens"].apply(len)
df["length_group"] = pd.qcut(df["n_words"].rank(method="first"), N_LENGTH_GROUPS, labels=False)
has_profanity = df["has_profanity"].to_numpy()
n_words = df["n_words"].to_numpy()
if len(aae_idx) < MIN_GROUP or len(main_idx) < MIN_GROUP:
    raise SystemExit(f"Too few 'neither' tweets in one group ({len(aae_idx)} AAE-aligned, {len(main_idx)} mainstream).")
print(f"{len(df)} 'neither' tweets: {len(aae_idx)} AAE-aligned, {len(main_idx)} mainstream")

def gap_with_ci(a, m):
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        return None
    gaps = [a[rng.integers(0, len(a), len(a))].mean() - m[rng.integers(0, len(m), len(m))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(gaps, [2.5, 97.5])
    return a.mean() - m.mean(), low, high

def p_aae_higher(a, m, n_boot=N_BOOTSTRAP_AUC):
    """P(AAE score > mainstream score) with a bootstrap CI (None if a group is too small)."""
    if len(a) < MIN_GROUP or len(m) < MIN_GROUP:
        return None
    def auc(x, y):
        return roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(y))], np.r_[x, y])
    values = [auc(a[rng.integers(0, len(a), len(a))], m[rng.integers(0, len(m), len(m))]) for _ in range(n_boot)]
    low, high = np.percentile(values, [2.5, 97.5])
    return auc(a, m), low, high

def print_rows(rows, flag, score):
    """rows: list of (name, boolean mask). Prints flag gap and P(AAE>main) for each subset."""
    print(f"{'subset':<26}{'n AAE':>6}{'n main':>7}{'flagged AAE':>12}{'AAE':>7}{'main':>7}{'gap':>8}   {'95% CI':<17}{'P(A>M)':>7}  95% CI")
    for name, mask in rows:
        a_idx, m_idx = mask & is_aae, mask & ~is_aae
        a, m = flag[a_idx], flag[m_idx]
        if a_idx.sum() < MIN_GROUP or m_idx.sum() < MIN_GROUP:
            print(f"{name:<26}{a_idx.sum():>6}{m_idx.sum():>7}   too few tweets in one group")
            continue
        gap, low, high = gap_with_ci(a, m)
        p, p_low, p_high = p_aae_higher(score[a_idx], score[m_idx])
        print(f"{name:<26}{a_idx.sum():>6}{m_idx.sum():>7}{int(a.sum()):>12}{a.mean():>7.3f}{m.mean():>7.3f}{gap:>+8.3f}   "
              f"[{low:+.3f}, {high:+.3f}]{p:>7.3f}  [{p_low:.3f}, {p_high:.3f}]")

def matched_gap(flag, strata, a_idx, m_idx):
    """AAE rate minus the mainstream rate re-weighted to the AAE mix of strata (only strata present in both groups)."""
    a_strata, m_strata = strata[a_idx], strata[m_idx]
    shared = [s for s in np.unique(a_strata) if (m_strata == s).any()]
    if not shared:
        return np.nan, 0.0
    keep = np.isin(a_strata, shared)
    weights = np.array([(a_strata == s).sum() for s in shared], dtype=float)
    main_rates = np.array([flag[m_idx][m_strata == s].mean() for s in shared])
    return flag[a_idx][keep].mean() - (weights * main_rates).sum() / weights.sum(), keep.mean()


print("\n=== 1. Profanity and length in each group ===")
for name, idx in [("AAE-aligned", aae_idx), ("mainstream", main_idx)]:
    print(f"{name:<12} n={len(idx):>6}   with profanity: {has_profanity[idx].mean():.3f}   median words: {np.median(n_words[idx]):.0f}")
print(f"{'words':<10}{'AAE-aligned':>13}{'mainstream':>13}   (% of each group)")
for low_n, high_n in LENGTH_BINS:
    in_bin = (n_words >= low_n) & (n_words <= high_n)
    label = f"{low_n}-{high_n}" if high_n < 1000 else f"{low_n}+"
    print(f"{label:<10}{in_bin[is_aae].mean() * 100:>13.1f}{in_bin[~is_aae].mean() * 100:>13.1f}")

matrices = {name: build_matrix(df[TEXT_COLUMN], config) for name, config in CONFIGS.items()}
X_main, words_main = matrices[MAIN_CONFIG]
for name, (X, words) in matrices.items():
    print(f"Vocabulary '{name}': {len(words)} words (min_df {CONFIGS[name]['min_df']})")

for key, model_name in MODELS:
    flag = df[f"flag_{key}"].to_numpy().astype(float)
    score = df[f"score_{key}"].to_numpy()
    observed = flag[is_aae].mean() - flag[~is_aae].mean()
    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)
    print(f"Flagged tweets among 'neither': AAE-aligned {int(flag[is_aae].sum())} of {is_aae.sum()}, mainstream {int(flag[~is_aae].sum())} of {(~is_aae).sum()}")
    if flag[is_aae].sum() < 15:
        print("WARNING: fewer than 15 flagged AAE-aligned tweets. Every flag-based number below is fragile; prefer P(A>M) and the CIs.")

    print("\n--- 2. With and without profanity ---")
    print_rows([("WITH profanity", has_profanity), ("WITHOUT profanity", ~has_profanity),
                ("all 'neither' tweets", np.ones(len(df), dtype=bool))], flag, score)

    print("\n--- 3. Inside bins of tweet length (words) ---")
    rows = []
    for low_n, high_n in LENGTH_BINS:
        label = f"{low_n}-{high_n} words" if high_n < 1000 else f"{low_n}+ words"
        rows.append((label, (n_words >= low_n) & (n_words <= high_n)))
    print_rows(rows, flag, score)

    print("\n--- 4. Gap after matching the groups (mainstream re-weighted to the AAE mix of strata) ---")
    strata_tercile = has_profanity.astype(int) * N_LENGTH_GROUPS + df["length_group"].to_numpy()
    strata_exact = np.minimum(n_words, MAX_WORDS) * 2 + has_profanity.astype(int)
    print(f"Observed gap: {observed:+.3f}")
    for label, strata in [("profanity x length terciles", strata_tercile), (f"exact word count (up to {MAX_WORDS}) x profanity", strata_exact)]:
        value, share = matched_gap(flag, strata, aae_idx, main_idx)
        boot = [matched_gap(flag, strata, rng.choice(aae_idx, len(aae_idx)), rng.choice(main_idx, len(main_idx)))[0] for _ in range(N_BOOTSTRAP)]
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        print(f"{label:<44}{value:>+8.3f}   95% CI [{low:+.3f}, {high:+.3f}]   (uses {share:.0%} of the AAE-aligned tweets)")

    print(f"\n--- 5A. Bag of words, flag at the default threshold ('{MAIN_CONFIG}' vocabulary, cross-validated) ---")
    flagged = flag.astype(int)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    def predict_flag(X, C):
        return cross_val_predict(LogisticRegression(C=C, max_iter=3000), X, flagged, cv=folds, method="predict_proba")[:, 1]
    losses = {C: log_loss(flagged, predict_flag(X_main, C)) for C in C_VALUES}
    best_C = min(losses, key=losses.get)
    print("Regularisation chosen by cross-validated log loss: " + "  ".join(f"C={C:g}: {loss:.4f}" for C, loss in losses.items()) + f"   -> C = {best_C:g}")
    predicted = predict_flag(X_main, best_C)
    predicted_gap = predicted[is_aae].mean() - predicted[~is_aae].mean()
    residual = flag - predicted
    boot = [residual[rng.choice(aae_idx, len(aae_idx))].mean() - residual[rng.choice(main_idx, len(main_idx))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{'':<22}{'AAE-aligned':>13}{'mainstream':>13}{'gap':>9}")
    print(f"{'observed flag rate':<22}{flag[is_aae].mean():>13.3f}{flag[~is_aae].mean():>13.3f}{observed:>+9.3f}")
    print(f"{'predicted from words':<22}{predicted[is_aae].mean():>13.3f}{predicted[~is_aae].mean():>13.3f}{predicted_gap:>+9.3f}")
    print(f"Residual (not explained by words): {residual[is_aae].mean() - residual[~is_aae].mean():+.3f}   95% CI [{low:+.3f}, {high:+.3f}]  (CI ignores the uncertainty of the fit)")
    if abs(observed) >= 0.02:
        print(f"Share of the observed gap reproduced by the words: {predicted_gap / observed:.0%}")
    else:
        print("The observed gap is too close to zero to express the reproduced part as a share.")
    print(f"AUC of the words-only model for the flag: {roc_auc_score(flagged, predicted):.3f}  (0.5 = chance)")
    print("Robustness of the residual to tokenizer and regularisation:")
    print(f"{'vocabulary':<12}{'C':>8}{'residual':>10}")
    for name, (X, _) in matrices.items():
        for C in C_VALUES:
            p = predict_flag(X, C)
            r = flag - p
            print(f"{name:<12}{C:>8g}{r[is_aae].mean() - r[~is_aae].mean():>+10.3f}")

    print(f"\n--- 5B. Bag of words, score without a threshold (ridge, '{MAIN_CONFIG}' vocabulary) ---")
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    def predict_score(X, y, alpha):
        return cross_val_predict(Ridge(alpha=alpha), X, y, cv=kfold)
    errors = {a: mean_squared_error(score, predict_score(X_main, score, a)) for a in ALPHA_VALUES}
    best_alpha = min(errors, key=errors.get)
    print("Regularisation chosen by cross-validated MSE: " + "  ".join(f"alpha={a:g}: {e:.4f}" for a, e in errors.items()) + f"   -> alpha = {best_alpha:g}")
    predicted_score = predict_score(X_main, score, best_alpha)
    observed_score_gap = score[is_aae].mean() - score[~is_aae].mean()
    predicted_score_gap = predicted_score[is_aae].mean() - predicted_score[~is_aae].mean()
    residual_score = score - predicted_score
    boot = []
    for _ in range(N_BOOTSTRAP_REFIT):
        sample = np.r_[rng.choice(aae_idx, len(aae_idx)), rng.choice(main_idx, len(main_idx))]
        r = score[sample] - predict_score(X_main[sample], score[sample], best_alpha)
        boot.append(r[is_aae[sample]].mean() - r[~is_aae[sample]].mean())
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{'':<22}{'AAE-aligned':>13}{'mainstream':>13}{'gap':>9}")
    print(f"{'observed mean score':<22}{score[is_aae].mean():>13.3f}{score[~is_aae].mean():>13.3f}{observed_score_gap:>+9.3f}")
    print(f"{'predicted from words':<22}{predicted_score[is_aae].mean():>13.3f}{predicted_score[~is_aae].mean():>13.3f}{predicted_score_gap:>+9.3f}")
    print(f"Residual (not explained by words): {residual_score[is_aae].mean() - residual_score[~is_aae].mean():+.3f}   95% CI [{low:+.3f}, {high:+.3f}]  (model refitted in {N_BOOTSTRAP_REFIT} resamples)")
    if abs(observed_score_gap) >= 0.02:
        print(f"Share of the observed score gap reproduced by the words: {predicted_score_gap / observed_score_gap:.0%}")
    else:
        print("The observed score gap is too close to zero to express the reproduced part as a share.")

print(f"\n=== 6. Words most tied to the AAE-aligned group among 'neither' tweets (words only; overall AAE share {is_aae.mean():.3f}) ===")
document_count = np.asarray(X_main.sum(axis=0)).ravel()
aae_count = np.asarray(X_main[is_aae].sum(axis=0)).ravel()
flag1 = df["flag_1"].to_numpy()
flag2 = df["flag_2"].to_numpy()
flagged_count = {"1": np.asarray(X_main[flag1 == 1].sum(axis=0)).ravel(), "2": np.asarray(X_main[flag2 == 1].sum(axis=0)).ravel()}
keep = document_count >= MIN_DOCUMENTS_LIST
share = np.where(keep, aae_count / np.maximum(document_count, 1), -1)
print(f"{'word':<16}{'tweets':>8}{'AAE share':>11}{'flag m1':>9}{'flag m2':>9}")
for i in np.argsort(-share)[:TOP_WORDS]:
    print(f"{words_main[i]:<16}{document_count[i]:>8}{share[i]:>11.3f}{flagged_count['1'][i] / document_count[i]:>9.3f}{flagged_count['2'][i] / document_count[i]:>9.3f}")