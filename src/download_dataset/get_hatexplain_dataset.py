"""
Download the HateXplain dataset (Mathew et al., 2021).
"""

import os
import json
import re
import urllib.request
from collections import Counter
import pandas as pd

URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
RAW_JSON_PATH = os.path.join("data", "raw", "hatexplain_dataset.json")
OUT_PATH = os.path.join("data", "raw", "hatexplain_twitter.csv")
LABEL_TO_CLASS = {"hatespeech": 0, "offensive": 1, "normal": 2}


print("-" * 30, " DOWNLOADING HATEXPLAIN DATASET (MATHEW ET AL., 2021) ", 30 * "-")

def tokens_to_text(tokens):
    """HateXplain replaces mentions and links with placeholders."""
    words = []
    for token in tokens:
        if token == "<user>":
            words.append("@user")
        elif token == "<url>":
            words.append("http")
        elif re.fullmatch(r"<\w+>", token):
            continue
        else:
            words.append(token)
    return " ".join(words)


os.makedirs(os.path.dirname(RAW_JSON_PATH), exist_ok=True)
if not os.path.exists(RAW_JSON_PATH):
    urllib.request.urlretrieve(URL, RAW_JSON_PATH)

with open(RAW_JSON_PATH, encoding="utf-8") as f:
    posts = json.load(f)

rows = []
n_undecided = 0
for post_id, post in posts.items():
    if not post_id.endswith("_twitter"):
        continue
    labels = Counter(a["label"] for a in post["annotators"])
    label, votes = labels.most_common(1)[0]
    if votes < 2:  # three different labels means that has no majority
        n_undecided += 1
        continue
    rows.append({"post_id": post_id, "tweet": tokens_to_text(post["post_tokens"]), "class": LABEL_TO_CLASS[label]})

df = pd.DataFrame(rows)
df = df[df["tweet"].str.strip() != ""]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)



print("--- Basic info ---")
print(f"Total tweets: {len(df)}")
print(f"Dropped Twitter posts without a majority label: {n_undecided}")

print("\n--- Class distribution ---")
print(df["class"].map({0: "hate speech", 1: "offensive language", 2: "neither"}).value_counts().to_string())

print("\n--- Class distribution (%) ---")
print((df["class"].map({0: "hate speech", 1: "offensive language", 2: "neither"}).value_counts(normalize=True) * 100).round(1).to_string())