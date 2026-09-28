# Dialectal Bias in Hate Speech Detection

## Overview
This is a project for Umeå University's NLP course (autumn 2026), investigating whether pretrained hate speech classifiers perform unfairly on text written in African-American English (AAE) compared to mainstream/standard English. It treats a widely deployed class of NLP system as the object of study, testing it for the kind of emergent bias described by Friedman and Nissenbaum (1996): a system that works well for the population it was built and tested on, but breaks down for others.

## Motivation
Hate speech classifiers, like the ones platforms such as X use to automatically flag or remove offensive content, are trained mostly on data written in "standard" English. Not everyone writes that way. AAE is a distinct, rule-governed dialect with its own grammar and vocabulary, not a "degraded" version of standard English. If a classifier learned its notion of "toxic" almost entirely from standard-English examples, it has no reliable way to know that AAE phrasing it has never seen isn't offensive either.

## Research Questions
- Does a pretrained hate-speech classifier show a significant performance gap between AAE-aligned text and mainstream-English text?
- Which words or stylistic patterns are most associated with false positives (AAE text wrongly flagged as hate speech)?
- Does fine-tuning the classifier's head on a balanced sample reduce this gap, and at what cost to overall performance?

## Approach
1. **Baseline.** Run a pretrained hate speech classifier on the test set with no modification, and split predictions by predicted dialect group.
2. **Dialect estimation.** Use the demographically-aligned dialect predictor from Blodgett et al. (2016) to estimate the probability that each tweet is AAE-aligned.
3. **Evaluation.** Compare precision, recall, and F1 between the AAE-aligned and mainstream-English groups, testing whether the gap is statistically significant with a bootstrap resampling test.
4. **Mitigation experiment.** Fine-tune only the classification head of the model (the underlying transformer stays frozen) on a class- and dialect-balanced subsample, then re-evaluate to see whether the gap narrows.

## Data
- **Source:** Davidson et al. (2017), 24,783 tweets labeled as `hate speech` / `offensive language` / `neither` by human CrowdFlower annotators.
- **Access:** full tweet text is included directly in the released CSV. (Waseem & Hovy (2016) was the original plan, but it's distributed as tweet-IDs-only, and the X/Twitter API no longer has a free tier for retrieving historical tweets.)
- **Dialect labels:** model-estimated probabilities (Blodgett et al., 2016), not self-reported by tweet authors.

See [`PROPOSAL.md`](./PROPOSAL.md) for the full project proposal, including hypothesis, assumptions, and expected outcome.

## Tech Stack
Python, Jupyter Notebooks, Hugging Face `transformers`, `scikit-learn`, `pandas`, `matplotlib`.

## Project Structure
```
├── data/
│   ├── raw/                    # original downloaded dataset
│   └── processed/              # dialect-scored data, model predictions
├── src/
│   ├── dialect_predictor/      # Blodgett et al. (2016) predictor, ported to Python 3
│   ├── 01_explore_dataset.py   # download + sanity-check the dataset
│   ├── 02_score_dialect.py     # estimate AAE-alignment per tweet
│   └── 03_evaluate_baseline.py # run the classifier, compare AAE vs. mainstream
├── results/
│   ├── figures/                # plots for the report/presentation
│   └── tables/                 # metrics tables
├── report/                     # final report draft
├── presentation/                # class presentation slides
└── scripts/                    # one-off utilities (e.g. GitHub project setup)
```

## Setup
```bash
git clone https://github.com/serac01/hate-speech-detection-system.git
cd hate-speech-detection-system
pip install -r requirements.txt
```
Run the pipeline in order, from the project root:
```bash
python src/01_explore_dataset.py
python src/02_score_dialect.py
python src/03_evaluate_baseline.py
```

## Educational Context
Developed for the NLP course at Umeå University (lecturer: Rahil, autumn 2026), as the course's final project.

## References
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.*
- Blodgett, S. L., Green, L., & O'Connor, B. (2016). *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.*
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). *The Risk of Racial Bias in Hate Speech Detection.*
- Davidson, T., Bhattacharya, D., & Weber, I. (2019). *Racial Bias in Hate Speech and Abusive Language Detection Datasets.*
- Bender, E. M. & Friedman, B. (2018). *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science.*