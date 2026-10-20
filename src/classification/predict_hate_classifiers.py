"""
Run the two pretrained classifiers on every tweet that has a dialect score, for the three datasets (Davidson, HateXplain and HateDay):
  1. cardiffnlp/twitter-roberta-base-hate           -> probability of "hate"      (trained on HatEval)
  2. unitary/unbiased-toxic-roberta (Detoxify)      -> probability of "toxicity"  (trained on Jigsaw Civil Comments)
For each model it saves the score of its positive label (hate_score_k) and the label at the default
threshold (pred_label_k = 1 if hate_score_k >= 0.5).
"""

import os
import json
import numpy as np
import pandas as pd

DATASETS = {
    "davidson": {"id": "Unnamed: 0",
                 "in": os.path.join("data", "processed", "davidson_with_dialect.csv"),
                 "out": os.path.join("data", "processed", "davidson_predictions.csv")},
    "hatexplain": {"id": "post_id",
                   "in": os.path.join("data", "processed", "hatexplain_with_dialect.csv"),
                   "out": os.path.join("data", "processed", "hatexplain_predictions.csv")},
    "hateday": {"id": "tweet_id",
                "in": os.path.join("data", "processed", "hateday_with_dialect.csv"),
                "out": os.path.join("data", "processed", "hateday_predictions.csv")},
}
MODELS = {
    "1": {"name": "cardiffnlp/twitter-roberta-base-hate", "backend": "transformers",
          "label": "hate", "preprocess": True},     # needs @user / http placeholders
    "2": {"name": "unitary/unbiased-toxic-roberta", "backend": "detoxify",
          "detoxify_name": "unbiased", "label": "toxicity", "preprocess": False},
}
BATCH_SIZE = 32
MAX_LENGTH = 128
RECORD_PATH = os.path.join("data", "processed", "classifier_models.json")  # which model produced each column

def preprocess(text):
    """Model 1 requirement: replace @mentions and URLs with placeholders."""
    new_text = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def score_with_transformers(model, texts):
    from transformers import pipeline
    # top_k=None returns the score of every class, so we can keep the probability of the positive label
    clf = pipeline("text-classification", model=model["name"], truncation=True, max_length=MAX_LENGTH, top_k=None)
    outputs = clf(texts, batch_size=BATCH_SIZE)
    return [next(c["score"] for c in o if c["label"] == model["label"]) for o in outputs]


def score_with_detoxify(model, texts):
    from detoxify import Detoxify
    detox = Detoxify(model["detoxify_name"])  # toxicity, severe_toxicity, obscene, ...
    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        scores.extend(detox.predict(texts[start:start + BATCH_SIZE])[model["label"]])
        if (start // BATCH_SIZE) % 100 == 0:
            print(f"  {start}/{len(texts)} tweets")
    return scores

# If a model changed, its columns are stale.
if os.path.exists(RECORD_PATH):
    with open(RECORD_PATH) as f:
        recorded = json.load(f)
else:
    recorded = {}

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
        if stale:
            print("Models changed since the saved predictions: " + ", ".join(f"{key} ({known.get(key)} -> {MODELS[key]['name']})" for key in stale))
            print("Their old predictions are discarded and recomputed.")
        previous = previous.drop_duplicates(id_column)
        for key in stale:  # drop the columns of models that changed
            previous = previous.drop(columns=[f"hate_score_{key}", f"pred_label_{key}"], errors="ignore")
        predictions = predictions.merge(previous, on=id_column, how="left")

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
            print(f"Running inference on {len(texts)} tweets...")
            scorer = score_with_detoxify if model["backend"] == "detoxify" else score_with_transformers
            scores = scorer(model, texts)
            predictions.loc[todo, score_column] = scores
            predictions.loc[todo, label_column] = (np.array(scores) >= 0.5).astype(int)
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        predictions.to_csv(cfg["out"], index=False)  # saved after each model, so a long run is not lost
        known[key] = model["name"]  # only now the columns of this model are known to be current
        with open(RECORD_PATH, "w") as f:
            json.dump(recorded, f, indent=1)

    for key in MODELS:
        predictions[f"pred_label_{key}"] = predictions[f"pred_label_{key}"].astype(int)
    predictions.to_csv(cfg["out"], index=False)
    print(f"\nSaved {len(predictions)} predictions to {cfg['out']}")
