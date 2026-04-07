"""
Debug script to identify which samples are misclassified
"""
import os
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join(os.path.dirname(__file__), "news.csv")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if df['label'].dtype == object:
        mapping = {'FAKE':1,'Fake':1,'fake':1,'REAL':0,'Real':0,'real':0}
        df['label'] = df['label'].map(mapping).fillna(0).astype(int)
    if 'title' in df.columns and df['title'].notna().any():
        texts = (df['title'].fillna('') + " - " + df['text'].fillna('')).tolist()
    else:
        texts = df['text'].fillna('').tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels, df

texts, labels, df = load_data(CSV_PATH)
train_texts, test_texts, train_labels, test_labels, train_idx, test_idx = train_test_split(
    texts, labels, range(len(texts)), test_size=0.2, random_state=1
)

vec = TfidfVectorizer(max_features=2000)
X_train = vec.fit_transform(train_texts)
X_test = vec.transform(test_texts)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)

print("=== MISCLASSIFIED SAMPLES ===\n")
for i, (true_label, pred_label, text_idx) in enumerate(zip(test_labels, preds, test_idx)):
    if true_label != pred_label:
        row = df.iloc[text_idx]
        label_name = "FAKE" if true_label == 1 else "REAL"
        pred_name = "FAKE" if pred_label == 1 else "REAL"
        print(f"Index {text_idx}: True={label_name}, Predicted={pred_name}")
        print(f"  Title: {row['title']}")
        print(f"  Text: {row['text'][:100]}...")
        print()

print("\n=== CORRECTLY CLASSIFIED SAMPLES ===\n")
correct_count = sum(1 for t, p in zip(test_labels, preds) if t == p)
print(f"Correct: {correct_count}/{len(test_labels)}")
