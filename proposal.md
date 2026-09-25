# Dialectal Bias in Hate Speech Detection Systems

**Sérgio Costa — NLP Final Project Proposal**
Umeå University, Autumn 2026

## Motivation
A researcher at Umeå University recently described a research misconduct complaint against her, based partly on AI-detection tools flagging her writing as AI-generated — tools she notes are "well documented... to be unreliable, and specifically biased against non-native English writers." Her case is a real example of the problem this project studies from another angle: language AI trained mostly on "standard" English misjudges anyone who writes outside that norm, whether a non-native speaker, or a speaker of a dialect like African-American English (AAE).

## Domain
Natural Language Processing — fairness and bias in text classification, specifically hate speech / toxicity detection.

## Scope
This project evaluates one pretrained hate speech classifier on one dataset, comparing its performance between two groups defined by predicted dialect (AAE-aligned vs. mainstream English). A balanced fine-tuning experiment is included as a secondary step to test whether the gap can be narrowed, time permitting.

## Main Hypothesis
Hate speech classifiers used on social media (e.g. X) are trained mostly on tweets in "standard" English. Not everyone writes that way — AAE, for example, is a distinct dialect with its own grammar and vocabulary. These classifiers perform worse, and more unfairly, on AAE text than on standard English.

## Research Questions / Objective
- Does a pretrained hate-speech classifier show a significant performance gap between AAE text and mainstream English text?
- Which words or patterns are most associated with false positives — wrongly flagging AAE text as hate speech?
- Does fine-tuning the model's classification head reduce this gap, and what does it cost in overall performance?

## Planned Approach
The project has four stages:

1. **Baseline.** Run an existing pretrained hate speech classifier on the test set, unmodified, to establish current performance before any intervention.
2. **Dialect estimation.** Use a probabilistic AAE-detection tool built by other researchers (Blodgett et al., 2016) to estimate, for each tweet, how likely it is to be written in AAE — a guess based on writing style, not something confirmed by the tweet's author.
3. **Stratified evaluation.** Split predictions into two groups — tweets likely written in AAE, and tweets likely written in standard English — and compare precision, recall, and F1 between them. Test whether any difference is statistically significant using a bootstrap resampling test.
4. **Mitigation experiment.** Fine-tune only the classifier's final classification layer (the underlying pretrained transformer stays frozen) on a sample balanced by both class (hate speech / not) and dialect group, using undersampling of the majority group rather than duplicating minority examples. Re-evaluate to see whether this narrows the performance gap, and at what cost to overall accuracy.

## Data
- **Dataset:** Waseem & Hovy (2016), Hate Speech Twitter Annotations — approximately 19,000 tweets labeled as hate speech or not.
- **Access:** the dataset is distributed as tweet IDs only, which must be "hydrated" (retrieved) via the platform's API; since the tweets date from 2013–2015, some may no longer be retrievable.
- **Dialect labels:** model-estimated probabilities (Blodgett et al., 2016), not self-reported identity.

## Assumptions About the Data
- Tweets were collected using deliberately provocative search terms to ensure a high rate of hate speech, so the sample skews toward extreme language rather than everyday tweets.
- AAE labels are model-estimated probabilities, not self-reported by tweet authors, and should be read as a proxy for AAE-associated language use rather than a claim about any individual's identity or background.
- No demographic information (age, gender, etc.) is available for the tweet authors.

## Limitations
- **Platform:** results are specific to Twitter/X and may not generalize to other platforms.
- **Temporal scope:** tweets date from 2013–2015; language use online has since evolved.
- **Sampling bias:** the dataset was curated to be rich in hate speech, not representative of everyday social media text.
- **Dialect estimation:** labels are probabilistic, not ground truth.
- **Class/group imbalance:** AAE-aligned examples are a minority in the dataset, which limits statistical power, particularly after balancing for the fine-tuning experiment.
- **Single dataset:** a second, model-labeled dataset (Davidson et al., 2017) was considered but excluded, since its labels come from a separate classifier with known label error (~91% accuracy) — using it would compound that dataset's labeling error with this project's own model error, confounding results.

## Expected Outcome
Based on prior research (Sap et al., 2019; Davidson et al., 2019), the baseline model is expected to perform worse on AAE-aligned tweets than on standard-English tweets, likely over-flagging AAE text as toxic due to surface-level overlap between AAE vocabulary and words the model associates with hate speech. Confirming this gap, and testing whether balanced fine-tuning of the classification head narrows it, would provide concrete evidence of dialectal bias in a widely deployed class of NLP system, along with a small-scale demonstration of one possible mitigation strategy.

## References
- Bender, E. M. & Friedman, B. (2018). *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science.* TACL, 6, 587–604.
- Blodgett, S. L., Green, L., & O'Connor, B. (2016). *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.* EMNLP.
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.
- Davidson, T., Bhattacharya, D., & Weber, I. (2019). *Racial Bias in Hate Speech and Abusive Language Detection Datasets.* Workshop on Abusive Language Online.
- Friedman, B. & Nissenbaum, H. (1996). *Bias in Computer Systems.* ACM Transactions on Information Systems, 14(3), 330–347.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). *The Risk of Racial Bias in Hate Speech Detection.* ACL.
- Waseem, Z. & Hovy, D. (2016). *Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter.* NAACL Student Research Workshop.