"""
Download the HateDay dataset (Tonneau et al., 2024) and keep the random sample of English tweets from the US.
"""

import os
import pandas as pd

HF_NAME = "manueltonneau/hateday"
LOCAL_FILES = [os.path.join("data", "raw", "hateday.parquet"), os.path.join("data", "raw", "hateday.csv")]
OUT_PATH = os.path.join("data", "raw", "hateday_us.csv")
COUNTRY = "US"
CLASS_TO_OURS = {2: 0, 1: 1, 0: 2}  # (map to the Davidson dataset) hateful -> hate speech, offensive -> offensive language, neutral -> neither
REQUIRED_COLUMNS = ["tweet_id", "text", "class_clean", "weighted", "lang_country_hateday"]

local_file = next((path for path in LOCAL_FILES if os.path.exists(path)), None)
if local_file is not None:
    print(f"Reading the local file {local_file}")
    df = pd.read_parquet(local_file) if local_file.endswith(".parquet") else pd.read_csv(local_file)
else:
    try:
        from datasets import load_dataset
        print(f"Downloading {HF_NAME} from Hugging Face...")
        df = load_dataset(HF_NAME, split="train").to_pandas()
    except Exception as error:
        raise SystemExit(f"Could not download {HF_NAME}: {error}\n"
                         "Accept the conditions on the dataset page, run `huggingface-cli login`, or download the file "
                         f"yourself and save it as one of: {', '.join(LOCAL_FILES)}")

missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    raise SystemExit(f"The file has no column(s) {missing}. Columns found: {list(df.columns)}")
print(f"{len(df)} rows in total")

df = df[(df["lang_country_hateday"] == COUNTRY) & (df["weighted"] == 0)].copy()
print(f"\n{len(df)} rows in the random sample of {COUNTRY}")

# Placeholders: HateDay writes @USER and LINK; the Davidson tweets (and what model 1 expects) use @user and http.
df["tweet"] = (df["text"].astype(str).str.replace(r"@USER", "@user", regex=True).str.replace(r"\bLINK\b", "http", regex=True))
df["class"] = df["class_clean"].map(CLASS_TO_OURS)
n_before = len(df)
df = df.dropna(subset=["class"])
df = df[df["tweet"].str.strip() != ""].drop_duplicates("tweet_id")
df["class"] = df["class"].astype(int)
print(f"Dropped {n_before - len(df)} rows without a valid label, with empty text or duplicated")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df[["tweet_id", "tweet", "class"]].to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} tweets to {OUT_PATH}")
print(df["class"].map({0: "hate speech", 1: "offensive language", 2: "neither"}).value_counts().to_string())
