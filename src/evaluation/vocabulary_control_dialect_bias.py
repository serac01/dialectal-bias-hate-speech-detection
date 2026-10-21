"""
Vocabulary control without a hand-made word list.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

DIALECT_PATH = os.path.join("data", "processed", "davidson_with_dialect.csv")
PREDICTIONS_PATH = os.path.join("data", "processed", "davidson_predictions.csv")
ID_COLUMN = "Unnamed: 0"
TEXT_COLUMN = "tweet"
MIN_DOCUMENTS_LIST = 100
TOP_WORDS = 25
N_FOLDS = 5
N_BOOTSTRAP = 2000        # CI of version A
N_BOOTSTRAP_REFIT = 100   # CI of version B (the model is refitted each time)
C_VALUES = [0.1, 1.0, 10.0, 100.0]   # inverse regularisation of the logistic regression
ALPHA_VALUES = [0.3, 1.0, 3.0, 10.0]  # regularisation of the ridge regression
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Tokenizers: "basic" is the earlier setting; "extended" keeps one-letter tokens (u, n, b), numbers and emojis.
CONFIGS = {"basic": {"token_pattern": r"[a-z][a-z']+", "min_df": 20, "emoji": False},
           "extended": {"token_pattern": r"[a-z0-9_']+", "min_df": 10, "emoji": True}}
MAIN_CONFIG = "extended"

def make_cleaner(keep_emoji):
    def clean(text):
        text = str(text).lower()
        if keep_emoji:
            text = re.sub(r"&#(\d+);", r" emoji\1 ", text)  # html entities of emojis become tokens
        text = re.sub(r"&#?\w+;", " ", text)                # other html entities such as &amp;
        text = re.sub(r"http\S+", " ", text)                # links
        text = re.sub(r"@\w+", " ", text)                   # mentions
        text = re.sub(r"\brt\b", " ", text)                 # retweet marker
        return text
    return clean

def build_matrix(texts, config):
    vectorizer = CountVectorizer(preprocessor=make_cleaner(config["emoji"]), token_pattern=config["token_pattern"], binary=True, min_df=config["min_df"])
    X = vectorizer.fit_transform(texts)
    return X, np.array(vectorizer.get_feature_names_out())

def folds_for(y):
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

def predict_flag(X, flagged, C):
    return cross_val_predict(LogisticRegression(C=C, max_iter=3000), X, flagged, cv=folds_for(flagged), method="predict_proba")[:, 1]

def predict_score(X, scores, alpha):
    return cross_val_predict(Ridge(alpha=alpha), X, scores, cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED))

def residual_gap(values, predicted, is_aae):
    residual = values - predicted
    return residual[is_aae].mean() - residual[~is_aae].mean()

df = pd.read_csv(DIALECT_PATH).merge(pd.read_csv(PREDICTIONS_PATH), on=ID_COLUMN).rename(columns={"pred_label_1": "flag_1", "pred_label_2": "flag_2", "hate_score_1": "score_1", "hate_score_2": "score_2"})
df["true_label"] = (df["class"] == 0).astype(int)  # 1 = hate speech, 0 = not hate
df = df[(df["dialect_group"].isin(["aae_aligned", "mainstream"])) & (df["true_label"] == 0)].reset_index(drop=True)
df["is_aae"] = (df["dialect_group"] == "aae_aligned").astype(int)
is_aae = df["is_aae"].to_numpy() == 1
aae_idx, main_idx = np.where(is_aae)[0], np.where(~is_aae)[0]

matrices = {name: build_matrix(df[TEXT_COLUMN], config) for name, config in CONFIGS.items()}
X_main, words_main = matrices[MAIN_CONFIG]

for key, model_name in [("1", "Model 1 (twitter-roberta-base-hate)"), ("2", "Model 2 (unbiased-toxic-roberta)")]:
    flagged = df[f"flag_{key}"].to_numpy()
    scores = df[f"score_{key}"].to_numpy()
    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)

    # 1. Words most linked to being flagged, and to the AAE-aligned group ---
    document_count = np.asarray(X_main.sum(axis=0)).ravel()
    flagged_count = np.asarray(X_main[flagged == 1].sum(axis=0)).ravel()
    aae_count = np.asarray(X_main[is_aae].sum(axis=0)).ravel()
    keep = document_count >= MIN_DOCUMENTS_LIST
    flag_rate_with_word = flagged_count / document_count
    aae_share_with_word = aae_count / document_count
    print(f"\n--- 1a. Words with the highest flag rate ---")
    print(f"{'word':<18}{'tweets':>8}{'flag rate':>11}{'AAE share':>11}")
    for i in np.argsort(-np.where(keep, flag_rate_with_word, -1))[:TOP_WORDS]:
        print(f"{words_main[i]:<18}{document_count[i]:>8}{flag_rate_with_word[i]:>11.3f}{aae_share_with_word[i]:>11.3f}")
    print(f"\n--- 1b. Words most tied to the AAE-aligned group ---")
    print(f"{'word':<18}{'tweets':>8}{'AAE share':>11}{'flag rate':>11}")
    for i in np.argsort(-np.where(keep, aae_share_with_word, -1))[:TOP_WORDS]:
        print(f"{words_main[i]:<18}{document_count[i]:>8}{aae_share_with_word[i]:>11.3f}{flag_rate_with_word[i]:>11.3f}")

    # --- 2A. The flag: can the words alone reproduce the gap? ---
    print(f"\n--- 2A. Flag at the default threshold (observed vs words-only predicted) ---")
    losses = {C: log_loss(flagged, predict_flag(X_main, flagged, C)) for C in C_VALUES}
    best_C = min(losses, key=losses.get)
    print("Regularisation chosen by cross-validated log loss: " + "  ".join(f"C={C:g}: {loss:.4f}" for C, loss in losses.items()) + f"   -> C = {best_C:g}")
    predicted = predict_flag(X_main, flagged, best_C)
    observed_gap = flagged[is_aae].mean() - flagged[~is_aae].mean()
    predicted_gap = predicted[is_aae].mean() - predicted[~is_aae].mean()
    print(f"{'':<22}{'AAE-aligned':>13}{'mainstream':>13}{'gap':>9}")
    print(f"{'observed flag rate':<22}{flagged[is_aae].mean():>13.3f}{flagged[~is_aae].mean():>13.3f}{observed_gap:>+9.3f}")
    print(f"{'predicted from words':<22}{predicted[is_aae].mean():>13.3f}{predicted[~is_aae].mean():>13.3f}{predicted_gap:>+9.3f}")
    residual = flagged - predicted
    boot = [residual[rng.choice(aae_idx, len(aae_idx))].mean() - residual[rng.choice(main_idx, len(main_idx))].mean() for _ in range(N_BOOTSTRAP)]
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{'residual (not explained by words)':<34}{residual_gap(flagged, predicted, is_aae):>+9.3f}   95% CI [{low:+.3f}, {high:+.3f}]  (CI ignores the uncertainty of the fit)")
    if abs(observed_gap) >= 0.05:
        print(f"Share of the observed gap reproduced by the words: {predicted_gap / observed_gap:.0%}")
    else:
        print("The observed gap is too close to zero to express the reproduced part as a share.")
    print(f"AUC of the words-only model for the classifier's flag: {roc_auc_score(flagged, predicted):.3f}  (0.5 = no better than chance)")

    # --- 2A'. Robustness of the residual to the tokenizer and the regularisation ---
    print("\nRobustness of the residual (version A) to tokenizer and regularisation:")
    print(f"{'vocabulary':<12}{'C':>8}{'residual':>10}{'reproduced':>12}")
    for name, (X, _) in matrices.items():
        for C in C_VALUES:
            p = predict_flag(X, flagged, C)
            gap_p = p[is_aae].mean() - p[~is_aae].mean()
            share = f"{gap_p / observed_gap:.0%}" if abs(observed_gap) >= 0.05 else "-"
            print(f"{name:<12}{C:>8g}{residual_gap(flagged, p, is_aae):>+10.3f}{share:>12}")

    # --- 2B. The score itself, without a threshold ---
    print(f"\n--- 2B. Score without a threshold: observed vs words-only predicted mean score (ridge, '{MAIN_CONFIG}' vocabulary) ---")
    errors = {a: mean_squared_error(scores, predict_score(X_main, scores, a)) for a in ALPHA_VALUES}
    best_alpha = min(errors, key=errors.get)
    print("Regularisation chosen by cross-validated MSE: " + "  ".join(f"alpha={a:g}: {e:.4f}" for a, e in errors.items()) + f"   -> alpha = {best_alpha:g}")
    predicted_score = predict_score(X_main, scores, best_alpha)
    observed_score_gap = scores[is_aae].mean() - scores[~is_aae].mean()
    predicted_score_gap = predicted_score[is_aae].mean() - predicted_score[~is_aae].mean()
    print(f"{'':<22}{'AAE-aligned':>13}{'mainstream':>13}{'gap':>9}")
    print(f"{'observed mean score':<22}{scores[is_aae].mean():>13.3f}{scores[~is_aae].mean():>13.3f}{observed_score_gap:>+9.3f}")
    print(f"{'predicted from words':<22}{predicted_score[is_aae].mean():>13.3f}{predicted_score[~is_aae].mean():>13.3f}{predicted_score_gap:>+9.3f}")
    boot = []
    for _ in range(N_BOOTSTRAP_REFIT):  # resample tweets, refit and predict again inside each resample
        sample = np.r_[rng.choice(aae_idx, len(aae_idx)), rng.choice(main_idx, len(main_idx))]
        boot.append(residual_gap(scores[sample], predict_score(X_main[sample], scores[sample], best_alpha), is_aae[sample]))
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{'residual (not explained by words)':<34}{residual_gap(scores, predicted_score, is_aae):>+9.3f}   95% CI [{low:+.3f}, {high:+.3f}]  (model refitted in {N_BOOTSTRAP_REFIT} resamples)")
    if abs(observed_score_gap) >= 0.02:
        print(f"Share of the observed score gap reproduced by the words: {predicted_score_gap / observed_score_gap:.0%}")
    else:
        print("The observed score gap is too close to zero to express the reproduced part as a share.")