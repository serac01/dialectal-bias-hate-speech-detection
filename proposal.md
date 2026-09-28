# Dialectal Bias in Hate Speech Detection Systems

**Sérgio Costa — NLP Final Project Proposal**
Umeå University, Autumn 2026

## Motivation
A researcher at Umeå University recently described a research misconduct complaint against her, based partly on AI-detection tools flagging her writing as AI-generated — tools she notes are "well documented... to be unreliable, and specifically biased against non-native English writers." Her case is a real example of the problem this project studies from another angle: language AI trained mostly on "standard" English misjudges anyone who writes outside that norm, whether a non-native speaker, or a speaker of a dialect like African-American English (AAE).

## Domain
Natural Language Processing — fairness and bias in text classification, specifically hate speech / toxicity detection.

## Scope
**Core:** evaluate one pretrained hate speech classifier on one dataset (Davidson et al., 2017), comparing its performance between two groups defined by predicted dialect (AAE-aligned vs. mainstream English), with a statistical significance test on the gap.

**Stretch (time permitting):** an error analysis of which words/patterns drive false positives, and a fine-tuning experiment on a balanced sample to test whether the gap narrows.

## Main Hypothesis
Hate speech classifiers used on social media (e.g. X) are trained mostly on tweets in "standard" English. Not everyone writes that way — AAE, for example, is a distinct dialect with its own grammar and vocabulary. These classifiers perform worse, and more unfairly, on AAE text than on standard English.

## Research Questions / Objective
- Does a pretrained hate-speech classifier show a significant performance gap between AAE text and mainstream English text?
- Which words or patterns are most associated with false positives — wrongly flagging AAE text as hate speech?
- Does fine-tuning the model's classification head reduce this gap, and what does it cost in overall performance?

## Planned Approach

### Model
The classifier evaluated is [`cardiffnlp/twitter-roberta-base-hate`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate), a RoBERTa-base model pretrained on ~58M tweets and fine-tuned on the **hate** subtask of the TweetEval benchmark (Barbieri et al., 2020) — which is the SemEval-2019 Task 5 (HatEval) dataset, targeting hate speech against **women and immigrants**, collected from Twitter around 2018-2019 (Basile et al., 2019).

**No train/test overlap:** this project's test data (Davidson et al., 2017) was collected independently, a year earlier, via a different method (scraping tweets matching a hate-speech lexicon, then CrowdFlower annotation) than HatEval's (targeted collection around women/immigrant-directed hate, EU/US annotators). The two datasets do not share a collection pipeline or time window, so there is no reason to expect overlap; this will be spot-checked by searching a sample of Davidson tweet texts against the public HatEval release.

**Label mapping:** the model outputs a binary label, `hate` or `not-hate`. Davidson et al.'s dataset has three classes: `hate_speech`, `offensive_language`, `neither`. This project maps `hate_speech` → `hate` (positive class) and `offensive_language` + `neither` → `not-hate`, following Davidson et al.'s own argument that offensive language and hate speech are analytically distinct categories.

**Important caveat this mapping introduces:** HatEval's notion of "hate" was defined specifically around women- and immigrant-directed hate, while Davidson et al.'s `hate_speech` label is broader (covers race, religion, sexual orientation, and disability, via the Hatebase.org lexicon). The classifier may therefore be systematically better or worse at recognizing hate speech outside the two categories it was trained on — this is noted as a limitation below, and is itself a source of measurement error separate from the dialect-bias question this project investigates.

### Stages
1. **Baseline.** Run the classifier above on the test set, unmodified, to establish current performance before any intervention.
2. **Dialect estimation.** Use a probabilistic AAE-detection tool built by other researchers (Blodgett et al., 2016) to estimate, for each tweet, how likely it is to be written in AAE — a guess based on writing style, not something confirmed by the tweet's author.
3. **Stratified evaluation.** Split predictions into two groups — tweets likely written in AAE, and tweets likely written in standard English — and compare precision, recall, and F1 between them. Test whether any difference is statistically significant using a bootstrap resampling test.
4. **(Stretch) Mitigation experiment.** Fine-tune only the classifier's final classification layer (the underlying pretrained transformer stays frozen) on a sample balanced by both class (hate speech / not) and dialect group, using undersampling of the majority group rather than duplicating minority examples. Re-evaluate to see whether this narrows the performance gap, and at what cost to overall accuracy.

## Data
- **Dataset:** Davidson et al. (2017), Automated Hate Speech Detection and the Problem of Offensive Language — 24,783 tweets, each labeled hate speech / offensive language / neither by majority vote of 3+ CrowdFlower human annotators.
- **Availability (verified):** the full tweet text is distributed directly in the dataset's public CSV ([github.com/t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)) — no API hydration is required, and no account, key, or payment is needed to access it. This has been downloaded and tested end-to-end already: all 24,783 rows load correctly with full tweet text present. (An earlier plan to use the Waseem & Hovy (2016) dataset was dropped for exactly this reason: it is distributed as tweet IDs only, and the X/Twitter API no longer offers a free tier for retrieving historical tweets by ID — pay-per-use pricing would cost roughly $95 for this dataset's size, with no guarantee all tweets are still available a decade later.)
- **Dialect labels:** model-estimated probabilities (Blodgett et al., 2016), not self-reported identity. Tweets with a predicted AAE-alignment score ≥0.6 are treated as "AAE-aligned" and ≤0.4 as "mainstream English"; scores in between are excluded as ambiguous. A stricter 0.8/0.2 threshold was tested first but left only 580 AAE-aligned tweets (29 labeled hate speech), too few for a reliable per-group comparison; 0.6/0.4 yields 4,689 AAE-aligned tweets (182 hate speech) and 13,211 mainstream tweets (992 hate speech), at the cost of somewhat less "pure" group separation.

## Assumptions About the Data
- Tweets were collected using a crowdsourced lexicon of hate speech and offensive terms, so the sample skews toward language likely to be hateful/offensive rather than everyday tweets.
- AAE labels are model-estimated probabilities, not self-reported by tweet authors, and should be read as a proxy for AAE-associated language use rather than a claim about any individual's identity or background.
- No demographic information (age, gender, etc.) is available for the tweet authors.

## Limitations
- **Label construct mismatch:** the classifier was trained to recognize hate speech targeting women and immigrants specifically (HatEval), while this project's ground truth (Davidson et al.) labels hate speech more broadly (race, religion, sexual orientation, disability, etc.). Any measured gap could partly reflect this mismatch rather than dialect alone; this is mitigated by comparing the *same* classifier across dialect groups on the *same* label definition, but is noted as a source of measurement noise.
- **Platform:** results are specific to Twitter/X and may not generalize to other platforms.
- **Temporal scope:** tweets date from 2017 data collection; language use online has since evolved.
- **Sampling bias:** the dataset was curated using a hate-speech lexicon, not representative of everyday social media text.
- **Dialect estimation:** labels are probabilistic, not ground truth; the 0.6/0.4 threshold trades some group purity for a large enough AAE-aligned sample to analyze.
- **Class/group imbalance:** even with the relaxed threshold, AAE-aligned examples remain a minority overall (4,689 vs. 13,211 mainstream), and hate-speech examples are a small fraction within each group (182 vs. 992), which limits statistical power, particularly after balancing for the fine-tuning experiment.
- **Single dataset:** results are tied to this dataset's specific curation choices (lexicon-based collection, CrowdFlower annotation guidelines) and may not generalize to hate speech data collected differently.

## Expected Outcome
Based on prior research (Sap et al., 2019; Davidson et al., 2019), the baseline model is expected to perform worse on AAE-aligned tweets than on standard-English tweets, likely over-flagging AAE text as toxic due to surface-level overlap between AAE vocabulary and words the model associates with hate speech. Confirming this gap, and testing whether balanced fine-tuning of the classification head narrows it, would provide concrete evidence of dialectal bias in a widely deployed class of NLP system, along with a small-scale demonstration of one possible mitigation strategy.

## References
- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP.
- Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P., & Sanguinetti, M. (2019). *SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter.* SemEval.
- Bender, E. M. & Friedman, B. (2018). *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science.* TACL, 6, 587–604.
- Blodgett, S. L., Green, L., & O'Connor, B. (2016). *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.* EMNLP.
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.
- Davidson, T., Bhattacharya, D., & Weber, I. (2019). *Racial Bias in Hate Speech and Abusive Language Detection Datasets.* Workshop on Abusive Language Online.
- Friedman, B. & Nissenbaum, H. (1996). *Bias in Computer Systems.* ACM Transactions on Information Systems, 14(3), 330–347.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). *The Risk of Racial Bias in Hate Speech Detection.* ACL.