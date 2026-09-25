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
- **Source:** Waseem & Hovy (2016), Hate Speech Twitter Annotations (~19,000 tweets, labeled hate speech / not).
- **Access:** distributed as tweet IDs only; text is retrieved by hydrating IDs through the platform's API. Some tweets may no longer be retrievable.
- **Dialect labels:** model-estimated probabilities (Blodgett et al., 2016), not self-reported by tweet authors.

See [`PROPOSAL.md`](./PROPOSAL.md) for the full project proposal, including hypothesis, assumptions, and expected outcome.

## Tech Stack
Python, Jupyter Notebooks, Hugging Face `transformers`, `scikit-learn`, `pandas`, `matplotlib`.

## Setup
```bash
git clone https://github.com/serac01/hate-speech-detection-system.git
cd hate-speech-detection-system
pip install -r requirements.txt
```
Notebooks are organized by stage: data collection and hydration, dialect estimation, baseline evaluation, and the fine-tuning experiment.

## Educational Context
Developed for the NLP course at Umeå University (lecturer: Rahil, autumn 2026), as the course's final project.

## References
- Waseem, Z. & Hovy, D. (2016). *Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter.*
- Blodgett, S. L., Green, L., & O'Connor, B. (2016). *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.*
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). *The Risk of Racial Bias in Hate Speech Detection.*
- Davidson, T., Bhattacharya, D., & Weber, I. (2019). *Racial Bias in Hate Speech and Abusive Language Detection Datasets.*
- Bender, E. M. & Friedman, B. (2018). *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science.*