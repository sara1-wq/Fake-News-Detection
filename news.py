import os
import sys
import pandas as pd
import numpy as np
import time
import matplotlib
# Prefer interactive GUI backend when running this script directly
if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except Exception as e:
        print(f"Could not set interactive matplotlib backend 'TkAgg': {e}")

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.linear_model import PassiveAggressiveClassifier,SGDClassifier
from sklearn.metrics import accuracy_score

sns.set(style='whitegrid')

def _safe_show(fname=None, save_default=True):
    """Save figure to a file and show it if backend is interactive.

    If `fname` is None and `save_default` is True, a timestamped filename will be generated.
    """
    backend = matplotlib.get_backend().lower()
    if fname is None and save_default:
        fname = f"figure_{int(time.time())}.png"

    if 'agg' in backend:
        if fname:
            plt.savefig(fname, bbox_inches='tight')
            print(f"Saved figure to {fname}")
        else:
            plt.close()
            print("Non-interactive backend active; figure not shown.")
    else:
        if fname:
            plt.savefig(fname, bbox_inches='tight')
            print(f"Saved figure to {fname}")
        plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

NEWS_CSV_PATH = r"C:\Users\Admin\Documents\news.csv"


def create_news_csv(path):
    # Create a small news CSV.
    news_df = pd.DataFrame({
        "title": [
            "Market crash Today",
            "Secret Cure Found",
            "Fed Raises Rates",
        ],
        "text": [
            "Stocks fell by 5% today following the news...",
            "Scientists found a fruit that cures every disease...",
            "The Federal Reserve announced a 0.25% hike...",
        ],
        "label": ["REAL", "Fake", "REAL"],
        "category": ["Business", "Health", "Economy"],
    })
    news_df.to_csv(path, index=False)
    print(f"News CSV written to {path}")


def load_news(path=NEWS_CSV_PATH, create_if_missing=False, text_candidates=("text", "title"), label_candidates=("category", "label")):
    # Load CSV; normalize text -> 'text', label -> 'category'
    if not os.path.exists(path):
        if create_if_missing:
            create_news_csv(path)
        else:
            raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    cols = set(df.columns)
    # find text and label columns
    text_col = next((c for c in text_candidates if c in cols), None)
    label_col = next((c for c in label_candidates if c in cols), None)

    # Handle missing columns gracefully by creating sensible defaults instead of raising an error
    if text_col is None:
        # Try to create 'text' by concatenating any object/string columns; fall back to the first column
        object_cols = [c for c in df.columns if df[c].dtype == object]
        if object_cols:
            df['text'] = df[object_cols].astype(str).apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        else:
            df['text'] = df.iloc[:, 0].astype(str)
        text_col = 'text'
        print("Warning: No text column found. Created 'text' by concatenating string columns or using the first column.")

    if label_col is None:
        # Create a default 'category' column when no label candidates are present
        df['category'] = 'Unknown'
        label_col = 'category'
        print("Warning: No label column found. Created 'category' with default value 'Unknown'.")

    # normalize columns
    if text_col != 'text':
        df = df.rename(columns={text_col: 'text'})
    if label_col != 'category':
        df = df.rename(columns={label_col: 'category'})

    return df


def main():
    # CLI: load CSV and show preview & counts
    import argparse
    parser = argparse.ArgumentParser(description='Load and preview news CSV')
    parser.add_argument('--csv', default=NEWS_CSV_PATH, help='Path to news CSV')
    parser.add_argument('--init', action='store_true', help='Create an example news CSV if missing')
    parser.add_argument('--preview', type=int, default=5, help='Number of rows to show (0 to skip)')
    args = parser.parse_args()

    # load CSV
    try:
        df = load_news(args.csv, create_if_missing=args.init)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)

    # show preview
    if args.preview > 0:
        print(df.head(args.preview))

    # show category counts or note missing
    if 'category' in df.columns:
        print('\nCategory counts:')
        print(df['category'].value_counts())
    else:
        print('\nNo "category" column found.')


if __name__ == '__main__':
    main()

# The code above provides functions to create, load, and preview a news CSV file with normalization of text and label columns.
# List of common English stopwords
STOP_WORDS =set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with"
])

def remove_stopwords(text):
    """Remove common stopwords from the input text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_words)

# Apply to DataFrame (demo)
if __name__ == '__main__':
    # Define df_final and apply cleaning for quick demo
    try:
        df_final = load_news(NEWS_CSV_PATH, create_if_missing=True)
    except Exception as e:
        print(f"Could not load news CSV: {e}")
        df_final = pd.DataFrame(columns=['title','text','category'])

    if not df_final.empty and 'text' in df_final.columns:
        df_final['cleaned_text'] = df_final['text'].apply(remove_stopwords)
        print("Original Text:", df_final['text'].iloc[0])
        print("Cleaned Text:", df_final['cleaned_text'].iloc[0])

        # Generate and display a word cloud from cleaned text
        try:
            all_text = ' '.join(df_final['cleaned_text'].astype(str))
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Cleaned Text')
            _safe_show()
        except Exception as e:
            print('WordCloud generation failed:', e)
    else:
        print("No data to clean.")

def train_and_evaluate(df):
    """Train a simple Naive Bayes model and show results. Expects 'cleaned_text' and either 'category' or 'label'."""
    # Ensure we have label column (prefer normalized 'category')
    if 'category' in df.columns:
        y = df['category']
    elif 'label' in df.columns:
        y = df['label']
    else:
        raise KeyError("No 'label' or 'category' column found in DataFrame")

    if 'cleaned_text' not in df.columns:
        raise KeyError("DataFrame must contain 'cleaned_text' column. Run preprocessing first.")

    # Initialize TF-IDF (removing common English filter words)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Convert your text column into a mathematical matrix
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model (Naive Bayes is great for text)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict the results for the test set
    y_pred = model.predict(X_test)

    # Show the 'Confusion Matrix'
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix : Performance Analysis')
    _safe_show()

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

# Get label column name (prefer 'category' over 'label')
label_col = 'category' if 'category' in df_final.columns else ('label' if 'label' in df_final.columns else None)
if label_col is None:
    print("No 'category' or 'label' column available for class-based word counts.")
else:
    # Get all words from FAKE news (if any)
    fake_rows = df_final[df_final[label_col] == 'Fake']
    if not fake_rows.empty and 'cleaned_text' in fake_rows.columns:
        fake_text = ' '.join(fake_rows['cleaned_text'].astype(str))
        # Simple word count logic
        from collections import Counter
        word_counts = Counter(fake_text.split())
        print("Top 5 words in FAKE news:", word_counts.most_common(5))
    else:
        print("No FAKE-labeled rows or no 'cleaned_text' available to compute word counts.")

# Save the word cloud as a high-quality PNG image
plt.savefig('my_fake_news_wordcloud.png', dpi=300, bbox_inches='tight')

# Prepare features X and target y  formodel training
X = df_final['cleaned_text']
y = df_final[label_col]

# Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initiate TF-IDF Vectorizer
max_df=0.7
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)

# Transform text into numerical Vectors
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize classifiers
sgd = SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=1000, tol=1e-3, random_state=0)
pac = PassiveAggressiveClassifier(max_iter=50)

# Fit PAC (simple online margin-based classifier)
pac.fit(tfidf_train, y_train)

# Calibrate SGD only when enough training samples exist for the chosen cv splits
n_train_samples = tfidf_train.shape[0]
cv_splits = min(5, n_train_samples)
if cv_splits < 2:
    print(f"Warning: Not enough training samples ({n_train_samples}) for cross-validation calibration; fitting SGD without calibration.")
    sgd.fit(tfidf_train, y_train)
    calibrated_sgd = sgd
else:
    calibrated_sgd = CalibratedClassifierCV(estimator=sgd, cv=cv_splits, method='sigmoid')
    try:
        calibrated_sgd.fit(tfidf_train, y_train)
    except Exception as e:
        print("Calibration failed, falling back to uncalibrated SGD:", e)
        sgd.fit(tfidf_train, y_train)
        calibrated_sgd = sgd

# Make predictions and calculate accuracy for both models
y_pred_pac = pac.predict(tfidf_test)
score_pac = accuracy_score(y_test, y_pred_pac)
print(f'PassiveAggressiveClassifier Accuracy: {round(score_pac*100,2)}%')

y_pred_sgd = calibrated_sgd.predict(tfidf_test)
score_sgd = accuracy_score(y_test, y_pred_sgd)
print(f'SGDClassifier (calibrated if possible) Accuracy: {round(score_sgd*100,2)}%')

# Create and plot confusion matrix for the chosen model (PAC by default)
cm = confusion_matrix(y_test, y_pred_pac, labels=['REAL', 'Fake'])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['REAL', 'Fake'], yticklabels=['REAL', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Final Project Result: Confusion Matrix (PAC)')
plt.show()

# Persist the trained models and vectorizer so the Streamlit app can load them
import pickle
MODEL_FILENAME = 'news_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
BASE = os.path.dirname(__file__)
model_path = os.path.join(BASE, MODEL_FILENAME)
vectorizer_path = os.path.join(BASE, VECTORIZER_FILENAME)

print('DEBUG: BASE=', BASE)
print('DEBUG: cwd=', os.getcwd())
print('DEBUG: base exists=', os.path.isdir(BASE), 'writable=', os.access(BASE, os.W_OK))

try:
    with open(model_path, 'wb') as mf:
        pickle.dump(pac, mf)
    with open(vectorizer_path, 'wb') as vf:
        pickle.dump(tfidf_vectorizer, vf)
    print(f"Saved model to {model_path} and vectorizer to {vectorizer_path}")
except Exception as e:
    import traceback
    print('Failed to save model/vectorizer, exception:')
    traceback.print_exc()




   






