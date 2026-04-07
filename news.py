from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sns.set(style="whitegrid")

# Default CSV path (relative)
NEWS_CSV_PATH = os.path.join(os.path.dirname(__file__), "news.csv")


def create_news_csv(path: str) -> None:
    """Create a small example CSV file with consistent 'text' and 'label' values."""
    df = pd.DataFrame(
        {
            "title": [
                "Market crash Today",
                "Secret Cure Found",
                "Fed Raises Rates",
                "Local Library Reopens",
                "Tech IPO Exceeds Estimates",
            ],
            "text": [
                "Stocks fell by 5% today following the news of several large sell orders.",
                "Researchers claim that a newly discovered compound cured disease X in lab tests — further study required.",
                "The Federal Reserve announced a 0.25% hike to interest rates and signaled a slow-down in purchases.",
                "The downtown library reopens after renovations; new safety features installed and events planned.",
                "Startup Z reported revenue exceeding estimates in its public filing, driving investor optimism.",
            ],
            # Use consistent uppercase labels that the Streamlit app expects
            "label": ["REAL", "FAKE", "REAL", "REAL", "REAL"],
            "category": ["Business", "Health", "Economy", "Local", "Business"],
        }
    )
    df.to_csv(path, index=False)
    print(f"Example CSV written to {path} (rows={len(df)})")


def _guess_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Pick sensible 'text' and 'label' columns from df, or create them if missing."""
    cols = set(df.columns)
    text_candidates = ["text", "body", "article", "content", "title"]
    label_candidates = ["label", "category", "target"]

    text_col = next((c for c in text_candidates if c in cols), None)
    label_col = next((c for c in label_candidates if c in cols), None)

    # If no text column, create one by concatenating object/string columns
    if text_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == object]
        if object_cols:
            df["text"] = df[object_cols].astype(str).apply(" ".join, axis=1)
        else:
            df["text"] = df.iloc[:, 0].astype(str)
        text_col = "text"
        print("Warning: no explicit text column found — created 'text' by concatenating string columns.")

    # If no label column, create 'label' with default 'UNKNOWN'
    if label_col is None:
        df["label"] = "UNKNOWN"
        label_col = "label"
        print("Warning: no label column found — created 'label' with default 'UNKNOWN'.")

    return text_col, label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series of labels into 'REAL' or 'FAKE' strings where possible.

    Rules:
    - Map common textual variants to REAL/FAKE.
    - If numeric {0,1} are present, map 1 -> 'FAKE', 0 -> 'REAL' (adjust if your dataset uses the opposite).
    - Otherwise uppercase and preserve original for inspection.
    """
    vals = series.dropna().unique()
    # numeric 0/1 mapping
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        try:
            uniq = set(int(x) for x in pd.Series(vals).astype(int).unique())
            if uniq <= {0, 1}:
                print("Detected numeric labels {0,1} — mapping: 1 -> 'FAKE', 0 -> 'REAL'.")
                return series.astype(int).map({1: "FAKE", 0: "REAL"})
        except Exception:
            pass

    def _map_one(x):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        if s in {"real", "true", "genuine", "trusted", "1"}:
            return "REAL"
        if s in {"fake", "false", "fraud", "hoax", "scam", "0"}:
            return "FAKE"
        return str(x).strip().upper()

    return series.apply(_map_one)


def load_news(path: str = NEWS_CSV_PATH, create_if_missing: bool = False) -> pd.DataFrame:
    """Load a CSV of news; return DataFrame with normalized 'text' and 'label' columns."""
    if not os.path.exists(path):
        if create_if_missing:
            create_news_csv(path)
        else:
            raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    text_col, label_col = _guess_text_and_label_columns(df)

    # Rename to canonical names
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df["text"] = df["text"].astype(str).fillna("")

    # Normalize labels
    df["label"] = normalize_labels(df["label"])

    # Drop rows missing text
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def preprocess_text(text: str) -> str:
    """Simple cleaning: remove URLs, excessive whitespace, and non-printable chars."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s\-\.,;:!?\']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save(
    df: pd.DataFrame,
    model_path: str = "news_model.pkl",
    vectorizer_path: str = "tfidf_vectorizer.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a Multinomial Naive Bayes classifier on df['text'] -> df['label'] and save model + vectorizer.
    Returns (model, vectorizer).
    """
    if "text" not in df.columns or "label" not in df.columns:
        raise KeyError("DataFrame must contain 'text' and 'label' columns")

    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Keep only known labels
    df_train = df[df["label"].isin({"REAL", "FAKE"})].reset_index(drop=True)
    if df_train.empty:
        raise ValueError("No rows with 'REAL' or 'FAKE' labels available for training.")

    X = df_train["clean_text"]
    y = df_train["label"]

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=test_size, random_state=random_state)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save artifacts
    with open(model_path, "wb") as mf:
        pickle.dump(model, mf)
    with open(vectorizer_path, "wb") as vf:
        pickle.dump(tfidf, vf)

    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return model, tfidf


def plot_confusion_matrix(y_true, y_pred, classes=("REAL", "FAKE"), fname: Optional[str] = None):
    """Plot a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if fname:
        plt.savefig(fname, bbox_inches="tight")
        print(f"Saved confusion matrix to {fname}")
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load, inspect, train and save a simple news classifier")
    parser.add_argument("--csv", default=NEWS_CSV_PATH, help="Path to news CSV")
    parser.add_argument("--init", action="store_true", help="Create a small example CSV at --csv if missing")
    parser.add_argument("--preview", type=int, default=0, help="Show N rows of the CSV (0 to skip)")
    parser.add_argument("--train", action="store_true", help="Train a classifier from the CSV")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate and print metrics after training")
    parser.add_argument("--save", action="store_true", help="Save trained model and vectorizer (default filenames used)")
    parser.add_argument("--model-path", default="news_model.pkl", help="Where to save the trained model (pickle)")
    parser.add_argument("--vectorizer-path", default="tfidf_vectorizer.pkl", help="Where to save the vectorizer (pickle)")
    args = parser.parse_args(argv)

    if args.init:
        create_news_csv(args.csv)

    try:
        df = load_news(args.csv, create_if_missing=args.init)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.preview and args.preview > 0:
        print(df.head(args.preview))

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False))
    else:
        print("\nNo label column found after normalization.")

    if args.train:
        model, tfidf = train_and_save(df, model_path=args.model_path, vectorizer_path=args.vectorizer_path)
        if args.evaluate:
            df_train = df[df["label"].isin({"REAL", "FAKE"})].reset_index(drop=True)
            df_train["clean_text"] = df_train["text"].apply(preprocess_text)
            X = tfidf.transform(df_train["clean_text"])
            y = df_train["label"].values
            y_pred = model.predict(X)
            print("\nOverall classification report on whole labeled set (not a proper test):")
            print(classification_report(y, y_pred))
            plot_confusion_matrix(y, y_pred, fname="confusion_matrix.png")


if __name__ == "__main__":
    main()


