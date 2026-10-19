"""
Download the Davidson et al. (2017) hate speech dataset and prints basic statistics
"""

import pandas as pd
import urllib.request
import os

URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
OUT_PATH = os.path.join("data", "raw", "davidson_dataset.csv")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

if not os.path.exists(OUT_PATH):
    urllib.request.urlretrieve(URL, OUT_PATH)

print("-" * 30, " DOWNLOADING DAVIDSON ET AL. (2017) HATE SPEECH DATASET ", 30 * "-")
df = pd.read_csv(OUT_PATH)

categories_map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
df["label"] = df["class"].map(categories_map)

print("--- Basic info ---")
print(f"Total tweets: {len(df)}")

print("\n--- Class distribution ---")
print(df["label"].value_counts())

print("\n--- Class distribution (%) ---")
print((df["label"].value_counts(normalize=True) * 100).round(1))