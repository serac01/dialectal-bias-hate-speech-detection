"""
Estimate how AAE-aligned each tweet is, with the dialect predictor of Blodgett et al. (2016),
for the three datasets (Davidson, HateXplain and HateDay).

For every tweet saves:
  - aae_score: first component of the 4-way posterior (African-American, Hispanic, Asian, White)
  - dialect_group: aae_aligned (score >= 0.6), mainstream (score <= 0.4), ambiguous (in between), unscored
  - the full posterior (p_aae, p_hispanic, p_asian, p_white) and the argmax group (aae_argmax / white_argmax / other_argmax)
  - aae_score_masked: the AAE score after REMOVING PROFANITY from the text before scoring.
    The groups are partly defined by the same words that make a classifier flag a tweet; scoring without
    them breaks that link. The tweet itself is not changed, only what the dialect model sees.
The score describes how much the wording resembles each group's usage, not who wrote the tweet.
"""

import os
import re
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # so that `blodgett` can be imported
from blodgett import predict, twokenize

DATASETS = {
    "davidson": (os.path.join("data", "raw", "davidson_dataset.csv"),
                 os.path.join("data", "processed", "davidson_with_dialect.csv")),
    "hatexplain": (os.path.join("data", "raw", "hatexplain_twitter.csv"),
                   os.path.join("data", "processed", "hatexplain_with_dialect.csv")),
    "hateday": (os.path.join("data", "raw", "hateday_us.csv"),
                os.path.join("data", "processed", "hateday_with_dialect.csv")),
}

# A tweet is "aae_aligned" above AAE_THRESHOLD, "mainstream" below MAINSTREAM_THRESHOLD
AAE_THRESHOLD = 0.6
MAINSTREAM_THRESHOLD = 1 - AAE_THRESHOLD
LABEL_NAMES = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
ARGMAX_NAMES = ["aae_argmax", "other_argmax", "other_argmax", "white_argmax"]  # AAE, Hispanic, Asian, White

# Profanity and slurs removed before scoring
PROFANITY = re.compile(
    r"^(b[i1]tch\w*|hoe?s?|hos|pussy|pussies|cunt\w*|slut\w*|whor\w*|skank\w*|nigg\w*|nicca\w*|nigah|"
    r"fuck\w*|shit\w*|ass|asses|dick\w*|fag\w*|retard\w*|damn\w*)$", re.IGNORECASE)

def assign_dialect_group(score):
    if pd.isna(score):
        return "unscored"
    if score >= AAE_THRESHOLD:
        return "aae_aligned"
    if score <= MAINSTREAM_THRESHOLD:
        return "mainstream"
    return "ambiguous"

def score_tweet(text):
    tokens = twokenize.tokenizeRawTweetText(str(text))
    posterior = predict.predict(tokens) # Returns a length-4 numpy array [African-American, Hispanic, Asian, White]
    posterior_masked = predict.predict([t for t in tokens if not PROFANITY.match(t)])
    return {
        "aae_score": np.nan if posterior is None else posterior[0],
        "p_aae": np.nan if posterior is None else posterior[0],
        "p_hispanic": np.nan if posterior is None else posterior[1],
        "p_asian": np.nan if posterior is None else posterior[2],
        "p_white": np.nan if posterior is None else posterior[3],
        "argmax_group": "unscored" if posterior is None else ARGMAX_NAMES[int(np.argmax(posterior))],
        "aae_score_masked": np.nan if posterior_masked is None else posterior_masked[0],
    }

predict.load_model()
for name, (in_path, out_path) in DATASETS.items():
    print("\n" + "-" * 30, f" DIALECT SCORES: {name.upper()} ", 30 * "-")
    if not os.path.exists(in_path):
        print(f"{in_path} not found: run the data stage first. Skipping...")
        continue
    df = pd.read_csv(in_path)
    
    print(f"Scoring {len(df)} tweets...")
    scores = pd.DataFrame([score_tweet(text) for text in df["tweet"]])
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df["dialect_group"] = df["aae_score"].apply(assign_dialect_group)
    df["label"] = df["class"].map(LABEL_NAMES)

    print("--- Dialect group sizes ---")
    print(df["dialect_group"].value_counts())
    print("\n--- Dialect group sizes by class label ---")
    print(pd.crosstab(df["dialect_group"], df["label"]))
    print("\n--- Argmax groups ---")
    print(df["argmax_group"].value_counts())
    print(f"\nTweets that cannot be scored: {df['aae_score'].isna().sum()}; after removing profanity: {df['aae_score_masked'].isna().sum()}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")