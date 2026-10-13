"""
Run the pretrained hate speech classifier (cardiffnlp/twitter-roberta-base-hate)
on the Davidson et al. tweets that score_aae_dialect.py assigned to a dialect group.
"""

import os
import pandas as pd
from transformers import pipeline

IN_PATH = os.path.join("data", "processed", "davidson_dataset_with_dialect.csv")
OUT_PATH = os.path.join("data", "processed", "davidson_dataset_with_dialect_and_roberta_classification.csv")
MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"

def preprocess(text):
    """Model requirement: replace @mentions and URLs with placeholders."""
    new_text = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


df = pd.read_csv(IN_PATH)
df = df[df["dialect_group"].isin(["aae_aligned", "mainstream"])].copy()
df["true_label"] = (df["class"] == 0).astype(int)  # Davidson ground truth: 1 = hate_speech, 0 = not hate

n_aae = (df["dialect_group"] == "aae_aligned").sum()
n_main = (df["dialect_group"] == "mainstream").sum()
print(f"Running on {len(df)} tweets ({n_aae} AAE-aligned, {n_main} mainstream)")

print(f"Loading {MODEL_NAME}...")
clf = pipeline("text-classification", model=MODEL_NAME, truncation=True, max_length=128)
texts = df["tweet"].apply(preprocess).tolist()

print(f"Running inference on {len(texts)} tweets...")
predictions = clf(texts, batch_size=32)

df["pred_label"] = [1 if p["label"] == "hate" else 0 for p in predictions]  # 1 = hate, 0 = not-hate
df["pred_score"] = [p["score"] for p in predictions]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved predictions to {OUT_PATH}")