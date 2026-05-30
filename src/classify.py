import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def train_text_classifier(
    train_df,
    label_col,
    text_col="full_text",
    max_features=30000,
):
    df = train_df[[text_col, label_col]].dropna()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    model.fit(df[text_col], df[label_col])
    return model


def evaluate_classifier(model, eval_df, label_col, text_col="full_text"):
    df = eval_df[[text_col, label_col]].dropna()

    y_true = df[label_col]
    y_pred = model.predict(df[text_col])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)