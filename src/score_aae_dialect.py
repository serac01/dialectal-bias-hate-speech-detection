"""
Estimate whether each tweet is written in AAE (African-American English) or mainstream English,
using the dialect predictor from Blodgett et al. (2016).
"""
print("\n\n\n", "-" * 30, " ESTIMATE WHETHER EACH TWEET IS WRITTEN IN AAE OR MAINSTREAM ENGLISH USING BLODGETT ET AL. PREDICTOR ", 30 * "-")

import os
import pandas as pd
from dialect_predictor import predict, twokenize


IN_PATH = os.path.join("data", "raw", "davidson_labeled_data.csv")
OUT_PATH = os.path.join("data", "processed", "tweets_with_dialect_group.csv")

# A tweet is "aae_aligned" above AAE_THRESHOLD, "mainstream" below MAINSTREAM_THRESHOLD
AAE_THRESHOLD = 0.6
MAINSTREAM_THRESHOLD = 1 - AAE_THRESHOLD

predict.load_model()
df = pd.read_csv(IN_PATH)

def score_tweet(text):
    tokens = twokenize.tokenizeRawTweetText(text)
    posterior = predict.predict(tokens) # Returns a length-4 numpy array [African-American, Hispanic, Asian, White]
    if posterior is None:
        return None
    return posterior[0]

def assign_dialect_group(score):
    if pd.isna(score):
        return "unscored"
    if score >= AAE_THRESHOLD:
        return "aae_aligned"
    if score <= MAINSTREAM_THRESHOLD:
        return "mainstream"
    return "ambiguous"


df["aae_score"] = df["tweet"].apply(score_tweet) # how close the tweet words are aligned to AAE
df["dialect_group"] = df["aae_score"].apply(assign_dialect_group)

print("--- Dialect group sizes ---")
print(df["dialect_group"].value_counts())

print("\n--- Dialect group sizes by class label ---")
label_map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
df["label"] = df["class"].map(label_map)
print(pd.crosstab(df["dialect_group"], df["label"]))

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")