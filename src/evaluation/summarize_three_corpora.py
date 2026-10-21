"""
One table with the main result for the three corpora and the two classifiers, on the same population
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CORPORA = [
    ("Davidson", "Unnamed: 0", "davidson"),
    ("HateXplain", "post_id", "hatexplain"),
    ("HateDay", "tweet_id", "hateday"),
]
MODELS = [("1", "twitter-roberta-base-hate"), ("2", "unbiased-toxic-roberta")]
DEFINITIONS = [("baseline", "dialect_group", "aae_aligned", "mainstream"),
               ("argmax", "argmax_group", "aae_argmax", "white_argmax")]
MAINSTREAM_FPRS = [0.10, 0.20]
MIN_GROUP = 30
N_BOOTSTRAP_GAP = 2000
N_BOOTSTRAP_AUC = 500
RANDOM_SEED = 42
OUT_DIR = "results"
rng = np.random.default_rng(RANDOM_SEED)


def fast_auc(a, m):
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(m))], np.r_[a, m])


rows = []
for corpus, id_column, stem in CORPORA:
    dialect_path = os.path.join("data", "processed", f"{stem}_with_dialect.csv")
    predictions_path = os.path.join("data", "processed", f"{stem}_predictions.csv")
    if not (os.path.exists(dialect_path) and os.path.exists(predictions_path)):
        print(f"{corpus}: files not found, skipped")
        continue
    full = pd.read_csv(dialect_path).merge(pd.read_csv(predictions_path), on=id_column)
    full = full[full["class"] == 2]
    for definition, column, aae_name, main_name in DEFINITIONS:
        is_aae = (full[column] == aae_name).to_numpy()
        is_main = (full[column] == main_name).to_numpy()
        if is_aae.sum() < MIN_GROUP or is_main.sum() < MIN_GROUP:
            print(f"{corpus} ({definition}): fewer than {MIN_GROUP} tweets in one group ({is_aae.sum()} / {is_main.sum()}), skipped")
            continue
        for key, model_name in MODELS:
            flag = full[f"pred_label_{key}"].to_numpy().astype(float)
            score = full[f"hate_score_{key}"].to_numpy()
            fa, fm = flag[is_aae], flag[is_main]
            gaps = [fa[rng.integers(0, len(fa), len(fa))].mean() - fm[rng.integers(0, len(fm), len(fm))].mean() for _ in range(N_BOOTSTRAP_GAP)]
            gap_low, gap_high = np.percentile(gaps, [2.5, 97.5])
            sa, sm = score[is_aae], score[is_main]
            auc = fast_auc(sa, sm)
            aucs = [fast_auc(sa[rng.integers(0, len(sa), len(sa))], sm[rng.integers(0, len(sm), len(sm))]) for _ in range(N_BOOTSTRAP_AUC)]
            auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])
            row = {"corpus": corpus, "definition": definition, "model": model_name, "n_aae": int(is_aae.sum()), "n_main": int(is_main.sum()),
                   "flagged_aae": int(fa.sum()), "fpr_aae": fa.mean(), "fpr_main": fm.mean(), "gap": fa.mean() - fm.mean(),
                   "gap_low": gap_low, "gap_high": gap_high, "p_aae_higher": auc, "p_low": auc_low, "p_high": auc_high}
            for target in MAINSTREAM_FPRS:
                threshold = np.quantile(sm, 1 - target)
                achieved = (sm >= threshold).mean()
                row[f"ratio_at_{int(target * 100)}"] = (sa >= threshold).mean() / achieved if achieved > 0 else np.nan
            rows.append(row)

table = pd.DataFrame(rows)
if table.empty:
    raise SystemExit("No corpus could be summarised: run the previous stages first.")
os.makedirs(OUT_DIR, exist_ok=True)
table.to_csv(os.path.join(OUT_DIR, "summary_three_corpora.csv"), index=False)

for definition, *_ in DEFINITIONS:
    subset = [r for r in rows if r["definition"] == definition]
    if not subset:
        continue
    print(f"\nTweets labelled 'neither', groups defined by: {definition}\n")
    print(f"{'corpus':<12}{'model':<27}{'n AAE':>6}{'n main':>8}{'flagged':>8}{'FPR AAE':>9}{'FPR main':>9}{'gap':>8}   {'95% CI':<17}{'P(A>M)':>7}  {'95% CI':<15}{'r@10%':>7}{'r@20%':>7}")
    for r in subset:
        print(f"{r['corpus']:<12}{r['model']:<27}{r['n_aae']:>6}{r['n_main']:>8}{r['flagged_aae']:>8}{r['fpr_aae']:>9.3f}{r['fpr_main']:>9.3f}{r['gap']:>+8.3f}   "
              f"[{r['gap_low']:+.3f}, {r['gap_high']:+.3f}]{r['p_aae_higher']:>7.3f}  [{r['p_low']:.3f}, {r['p_high']:.3f}]{r['ratio_at_10']:>7.2f}{r['ratio_at_20']:>7.2f}")
print("\n'flagged' = number of AAE tweets flagged: with fewer than about 15 the gap rests on a handful of tweets.")
print("P(A>M) is comparable across corpora; absolute gaps are not (they depend on each corpus's base rate of flags).")

lines = [r"\begin{tabular}{lllrrrrrr}", r"\hline",
         r"Corpus & Groups & Model & $n_{AAE}$ & FPR AAE & FPR main. & Gap [95\% CI] & $P(\mathrm{AAE}>\mathrm{main.})$ & Ratio@10\% \\", r"\hline"]
for r in rows:
    lines.append(f"{r['corpus']} & {r['definition']} & {r['model']} & {r['n_aae']} & {r['fpr_aae']:.3f} & {r['fpr_main']:.3f} & "
                 f"{r['gap']:+.3f} [{r['gap_low']:+.3f}, {r['gap_high']:+.3f}] & {r['p_aae_higher']:.3f} & {r['ratio_at_10']:.2f} \\\\")
lines += [r"\hline", r"\end{tabular}"]
with open(os.path.join(OUT_DIR, "summary_three_corpora.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved {OUT_DIR}/summary_three_corpora.csv and {OUT_DIR}/summary_three_corpora.tex")
