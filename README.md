# Dialectal Bias in Hate Speech Detection

## Overview
This is a project for Umeå University's NLP course (autumn 2026). It **audits** a pretrained hate speech classifier, [`cardiffnlp/twitter-roberta-base-hate`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate), to see whether it treats tweets written in African-American English (AAE)-aligned wording differently from mainstream-English tweets. The model is run as is: nothing is trained or fine-tuned.

The study looks for the kind of emergent bias described by Friedman and Nissenbaum (1996): a system that works for the population it was built and tested on, but breaks down for others. The classifier is **not** the system any platform (such as X) actually runs. It is an open model used as a representative of this class of systems.

## Motivation
Hate speech classifiers are trained mostly on data in "standard" English. AAE is a distinct, rule-governed variety of English with its own grammar and vocabulary, not a degraded version of standard English. A classifier that learned "toxic" from mostly standard-English examples may flag AAE-aligned text that is not hateful (Sap et al., 2019; Davidson et al., 2019).

## Research Questions
- **RQ1.** Does the classifier show a significant performance gap between AAE-aligned and mainstream-English tweets?
- **RQ2.** What explains the gap? Which words and label types are most associated with the false positives?
- **RQ3 (future work).** Does fine-tuning the classification head on a balanced sample reduce the gap, and at what cost? Not done in this project.

## Pipeline
Three components with different roles. They are independent models, but the pipeline is sequential: each step reads the previous step's output.

| Component | Role |
|---|---|
| Davidson et al. (2017) | Ground truth. Labels: hate speech / offensive language / neither. |
| Blodgett et al. (2016) dialect predictor | Measurement. Gives each tweet an AAE-alignment score (first component of its 4-way posterior). |
| `twitter-roberta-base-hate` | The audited classifier. Binary: hate / not hate. |

```
Davidson tweets -> dialect score (Blodgett) -> group assignment -> RoBERTa predictions -> evaluation + bootstrap
```

Key decisions:
- **Label mapping.** Davidson `hate_speech` = positive. `offensive_language` and `neither` = not hate. The classifier is trained on a narrower definition of hate than Davidson's, which matters for the results below.
- **Groups.** AAE-aligned: score >= 0.6. Mainstream: score <= 0.4. Ambiguous tweets (~6,900) and unscored tweets (7) are excluded. **17,900 tweets remain: 4,689 AAE-aligned and 13,211 mainstream.**
- **Inference.** Max length 128, `@user` / `http` preprocessing, default decision threshold.
- **Evaluation.** Precision, recall, F1, accuracy, false positive rate (FPR), plus context numbers (prevalence, share flagged, precision lift). Gaps are AAE minus mainstream, with a paired-group bootstrap (2,000 resamples, seed 42, 95% percentile CI).

## Main results

## Limitations

## Data
- **Source.** Davidson et al. (2017). The released CSV has 24,783 tweets labelled `hate speech` (1,430), `offensive language` (19,190) or `neither` (4,163) by CrowdFlower annotators (the paper reports 24,802 labelled; the 19 difference is unexplained).
- **Access.** The tweet text is in the released CSV. Waseem & Hovy (2016) was considered, but it is distributed as IDs only and the X/Twitter API no longer gives free access to historical tweets.
- **Dialect labels.** Model-estimated, not self-reported.
- The report contains a data statement following Bender and Friedman (2018).

## Project Structure
```
├── src/
│   ├── dialect_predictor/
│   ├── get_davidson_dataset.py
│   ├── score_aae_dialect.py
│   ├── predict_roberta.py
│   ├── evaluate_roberta.py
├── report/
│   ├── report.tex, 
│   ├── report.pdf
│   └── references.bib
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/serac01/hate-speech-detection-system.git
cd hate-speech-detection-system
python -m pip install -r requirements.txt
```

## References