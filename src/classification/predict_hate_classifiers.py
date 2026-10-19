"""
Run the two pretrained hate speech classifiers on every tweet that has a dialect score, for both datasets:
  1. cardiffnlp/twitter-roberta-base-hate
  2. facebook/roberta-hate-speech-dynabench-r4-target
For each model it saves the probability of the hate class (hate_score_k) and the predicted label
at the default threshold (pred_label_k = 1 if hate_score_k >= 0.5).
"""

import os
import numpy as np
import pandas as pd
from transformers import pipeline

DATASETS = {
    "davidson": {"id": "Unnamed: 0",
                 "in": os.path.join("data", "processed", "davidson_with_dialect.csv"),
                 "out": os.path.join("data", "processed", "davidson_predictions.csv")},
    "hatexplain": {"id": "post_id",
                   "in": os.path.join("data", "processed", "hatexplain_with_dialect.csv"),
                   "out": os.path.join("data", "processed", "hatexplain_predictions.csv")},
}
MODELS = {
    "1": {"name": "cardiffnlp/twitter-roberta-base-hate", "preprocess": True},    # needs @user / http placeholders
    "2": {"name": "facebook/roberta-hate-speech-dynabench-r4-target", "preprocess": False},
}
HATE_LABEL = "hate"
BATCH_SIZE = 32
MAX_LENGTH = 128

def preprocess(text):
    """Model 1 requirement: replace @mentions and URLs with placeholders."""
    new_text = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

for dataset, cfg in DATASETS.items():
    id_column = cfg["id"]
    print("\n" + "-" * 30, f" CLASSIFIERS: {dataset.upper()} ", 30 * "-")
    if not os.path.exists(cfg["in"]):
        print(f"{cfg['in']} not found: run the dialect stage first. Skipping.")
        continue
    df = pd.read_csv(cfg["in"])
    df = df[df["dialect_group"] != "unscored"].reset_index(drop=True)

    if os.path.exists(cfg["out"]):
        previous = pd.read_csv(cfg["out"])
        print(f"Found {len(previous)} saved predictions.")
    else:
        print("No predictions file yet.")
    predictions = df[[id_column]].copy()  # one row per tweet, same order as df
    if previous is not None:
        predictions = predictions.merge(previous.drop_duplicates(id_column), on=id_column, how="left")

    for key, model in MODELS.items():
        score_column, label_column = f"hate_score_{key}", f"pred_label_{key}"
        if score_column not in predictions:
            predictions[score_column], predictions[label_column] = np.nan, np.nan
        todo = predictions[score_column].isna().to_numpy()
        print(f"\nModel {key} ({model['name']}): {int((~todo).sum())} tweets already done, {int(todo.sum())} to run.")
        if todo.sum() > 0:
            texts = df.loc[todo, "tweet"].astype(str)
            texts = texts.apply(preprocess).tolist() if model["preprocess"] else texts.tolist()
            print(f"Loading {model['name']}...")
            # top_k=None returns the score of every class, so we can keep the probability of "hate"
            clf = pipeline("text-classification", model=model["name"], truncation=True, max_length=MAX_LENGTH, top_k=None)
            print(f"Running inference on {len(texts)} tweets...")
            outputs = clf(texts, batch_size=BATCH_SIZE)
            scores = [next(c["score"] for c in o if c["label"] == HATE_LABEL) for o in outputs]
            predictions.loc[todo, score_column] = scores
            predictions.loc[todo, label_column] = (np.array(scores) >= 0.5).astype(int)
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        predictions.to_csv(cfg["out"], index=False)  # saved after each model, so a long run is not lost

    for key in MODELS:
        predictions[f"pred_label_{key}"] = predictions[f"pred_label_{key}"].astype(int)
    predictions.to_csv(cfg["out"], index=False)
    print(f"\nSaved {len(predictions)} predictions to {cfg['out']}")
