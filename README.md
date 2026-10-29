# Dialectal Bias in Hate Speech Detection

NLP course project, Umeå University, autumn 2026. By Sérgio Costa.

This project audits two **pretrained** hate speech classifiers. It asks whether they flag tweets that are *not* hate more often when the tweets are written in a way that resembles African-American English (AAE). **No model is trained or fine-tuned**: both classifiers are used exactly as released, at their default threshold of 0.5.

## Research questions

1. Do the classifiers flag AAE-aligned tweets that are not hate more often than mainstream tweets?
2. What explains the gap: the mix of labels, profanity and women-directed terms, tweet length, or vocabulary?

The proposal also asked whether fine-tuning reduces the gap. This was moved to future work (see [Limitations](#limitations-and-future-work)).

## What the pipeline does

| Step | Script folder | What it does |
|---|---|---|
| 1. Data | `src/download_dataset/` | Downloads Davidson et al. (2017), the Twitter part of HateXplain, and the US random sample of HateDay. All three are mapped to the same labels: hate speech, offensive language, neither. |
| 2. Dialect | `src/score_dialect/` | Scores every tweet with the dialect model of Blodgett et al. (2016). A tweet is `aae_aligned` for a score of 0.6 or more and `mainstream` for 0.4 or less. Other groupings (argmax, 0.7/0.3, 0.8/0.2, 0.9/0.1, profanity removed before scoring) are used as robustness checks. |
| 3. Classify | `src/classification/` | Runs the two classifiers and saves each score and the label at threshold 0.5. |
| 4. Evaluate | `src/evaluation/` | Measures the gap between the two groups and tests explanations (see below). |

The dialect score measures how much the **words** of a tweet resemble each group's usage. It does not identify the author.

### Classifiers

| | Model | Detects | Trained on | Positive class used here |
|---|---|---|---|---|
| 1 | `cardiffnlp/twitter-roberta-base-hate` | hate speech | HatEval tweets (2018) | hate speech |
| 2 | `unitary/unbiased-toxic-roberta` (Detoxify "unbiased") | toxicity | Jigsaw comments (not tweets) | hate speech or offensive language |

For model 1 the false positives are tweets that are not hate. For model 2 they are only the tweets labelled *neither*, because offensive tweets are real positives for a toxicity model. A first candidate for model 2 was replaced because its training data included Davidson tweets.

### Measures

- **FPR gap**: share of tweets labelled *neither* that get flagged, AAE-aligned minus mainstream, with a 95% bootstrap CI (2000 resamples, seed 42).
- **P(AAE score > mainstream score)**: threshold-free (AUC). 0.5 means no difference. Comparable across corpora; absolute gaps are not.
- **FPR ratio** at a fixed mainstream FPR of 10% and 20%.
- **Controls** for RQ2: label mix (gap standardised on the label mix), women-directed profanity, a bag-of-words vocabulary model (cross-validated), profanity and tweet length on HateDay, and other group definitions.
- Groups with fewer than 30 tweets are not reported.

## Results (tweets labelled "neither", baseline groups)

| Corpus | Model | Gap [95% CI] | P(AAE > mainstream) |
|---|---|---|---|
| Davidson | 1 hate | +0.078 [0.025, 0.142] | 0.674 |
| Davidson | 2 toxicity | +0.091 [0.002, 0.180] | 0.656 |
| HateXplain | 1 hate | +0.248 [0.188, 0.308] | 0.749 |
| HateXplain | 2 toxicity | +0.304 [0.249, 0.357] | 0.700 |
| HateDay (US) | 1 hate | +0.024 [0.006, 0.045] | 0.733 |
| HateDay (US) | 2 toxicity | +0.065 [0.031, 0.103] | 0.668 |

- The gap is positive in all three corpora and for both models.
- On Davidson, F1 is almost identical for both groups (0.058 vs 0.056), while the FPR on non-hate tweets is 0.751 vs 0.338 (gap +0.412). About one third of this gap disappears when the label mix is matched (+0.281).
- A words-only model reproduces 89% (model 1) and 97% (model 2) of the Davidson gap, but only 12% and 21% on HateDay.
- Model 1 on HateDay rests on only 9 flagged AAE-aligned tweets, so it is fragile.

The full table is written to `results/summary_three_corpora.csv` and `results/summary_three_corpora.tex`.

## Repository structure

```
.
├── pipeline.ipynb                # runs every step in order
├── src/
│   ├── blodgett/                 # dialect model of Blodgett et al. (Python 3 port): predict.py, twokenize.py, model files
│   ├── download_dataset/         # get_davidson_dataset.py, get_hatexplain_dataset.py, get_hateday_dataset.py
│   ├── score_dialect/            # score_dialect_blodgett.py
│   ├── classification/           # predict_hate_classifiers.py
│   └── evaluation/               # baseline, per-label, threshold-free, controls, robustness, per-corpus and summary scripts
├── report/                       # final report
├── proposal.md
├── requirements.txt
└── README.md
```

`data/` (raw and processed files) and `results/` are created when the pipeline runs. Data files are not committed.

## Setup

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The first run downloads the two classifiers from Hugging Face. The Davidson and HateXplain data come from their public GitHub repositories.

### HateDay (gated dataset)

HateDay requires access approval on Hugging Face (`manueltonneau/hateday`).

1. Accept the conditions on the dataset page.
2. Log in with `huggingface_hub.login()` (or `huggingface-cli login`), or download the file yourself and save it as `data/raw/hateday.parquet` or `data/raw/hateday.csv`.

Never commit your token. HateDay tweets must not be committed, printed or quoted in documents. Offensive words are masked in the report.

## Run

Open `notebooks/pipeline.ipynb` and run the cells in order. A step is skipped if its output already exists; set `FORCE = True` in the first code cell to redo everything. Alternatively, run the scripts one by one from the repository root, in the order of the steps above:

```bash
python src/download_dataset/get_davidson_dataset.py
python src/score_dialect/score_dialect_blodgett.py
python src/classification/predict_hate_classifiers.py
python src/evaluation/baseline_evaluation.py
# ... and the other scripts in src/evaluation/
python src/evaluation/summarize_three_corpora.py
```

Scripts use relative paths, so run them from the repository root. Running the classifiers on CPU can take a while.

## Limitations and future work

- Only false positives are audited. Recall cannot be tested (only 2 hate tweets in the HateDay AAE-aligned group).
- The groups come from words, so dialect bias cannot be separated from lexical bias.
- Few AAE-aligned tweets are labelled *neither* in Davidson (95), so intervals are wide.
- Toxicity (model 2) is not the same as hate speech.
- The test data is independent of the models' training data by origin (other collections and years), not checked tweet by tweet. Duplicates and overlap were not checked.
- **Future work:** mitigation by fine-tuning with a fixed definition of hate (moved out of this project after discussion with the lecturer, because fine-tuning on Davidson would change what the classifier counts as hate); masking the flagged words and rerunning; the same text written in AAE and in mainstream English; duplicate and overlap checks.

## Changes since the proposal

- A second classifier was added after feedback.
- Davidson et al. replaced Waseem and Hovy, which is distributed as tweet IDs only.
- HateXplain (Twitter) and HateDay were added as extra corpora.
- Fine-tuning was moved to future work.

## References

- Davidson, T., Warmsley, D., Macy, M., Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM.
- Blodgett, S. L., Green, L., O'Connor, B. (2016). Demographic Dialectal Variation in Social Media: A Case Study of African-American English. EMNLP.
- Sap, M., Card, D., Gabriel, S., Choi, Y., Smith, N. A. (2019). The Risk of Racial Bias in Hate Speech Detection. ACL.
- Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., Mukherjee, A. (2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. AAAI.
- Tonneau, M., et al. (2024). HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter.