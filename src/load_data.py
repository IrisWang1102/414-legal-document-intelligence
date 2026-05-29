import pandas as pd
from datasets import load_dataset
import os

DATASET_NAME = "allenai/multi_lexsum"
DATASET_CONFIG = "v20230518"


def load_multilexsum():
    return load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)


def build_case_text(row):
    return "\n\n".join(row["sources"])


def extract_case_metadata(row):
    meta = row["case_metadata"]
    return pd.Series({
        "case_name": meta.get("case_name"),
        "court": meta.get("court"),
        "date_filed": meta.get("date_filed"),
        "case_type": meta.get("case_type"),
        "class_action_sought": meta.get("class_action_sought"),
    })


def build_clean_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    df = split_df.copy()

    df["full_text"] = df.apply(build_case_text, axis=1)

    metadata_df = df.apply(extract_case_metadata, axis=1)
    df = pd.concat([df, metadata_df], axis=1)

    df["text_length"] = df["full_text"].str.len()

    keep_cols = [
        "id",
        "case_name",
        "court",
        "date_filed",
        "case_type",
        "class_action_sought",
        "summary/long",
        "summary/short",
        "summary/tiny",
        "full_text",
        "text_length",
    ]

    return df[keep_cols]


def save_processed_splits(output_dir="data/processed"):
    dataset = load_multilexsum()

    for split in ["train", "validation", "test"]:
        raw_df = dataset[split].to_pandas()
        clean_df = build_clean_dataframe(raw_df)
        os.makedirs(output_dir, exist_ok=True)
        clean_df.to_csv(os.path.join(output_dir, f"{split}_clean.csv"), index=False)

    print("Processed splits saved successfully.")


if __name__ == "__main__":
    save_processed_splits()