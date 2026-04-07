import os
import random
import argparse
import pandas as pd  # type: ignore
import torch

# parse a quick flag to allow running a lightweight mock evaluation
parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true', help='Run a lightweight sklearn-based mock evaluation (no transformers)')
args = parser.parse_args()

# only import transformers when not running mock mode (speeds up quick checks)
if not args.mock:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Try to import sklearn metrics; fallback to simple implementations if missing
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
except Exception:
    accuracy_score = None
    precision_recall_fscore_support = None
    confusion_matrix = None
    train_test_split = None

CSV_PATH = os.path.join(os.path.dirname(__file__), "news.csv")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # normalize labels if strings
    if df['label'].dtype == object:
        mapping = {'FAKE':1,'Fake':1,'fake':1,'REAL':0,'Real':0,'real':0}
        df['label'] = df['label'].map(mapping).fillna(0).astype(int)
    # build text field
    if 'title' in df.columns and df['title'].notna().any():
        texts = (df['title'].fillna('') + " - " + df['text'].fillna('')).tolist()
    else:
        texts = df['text'].fillna('').tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

def simple_split(texts, labels, test_size=0.2, seed=1):
    combined = list(zip(texts, labels))
    random.Random(seed).shuffle(combined)
    n_test = max(1, int(len(combined) * test_size))
    test = combined[:n_test]
    train = combined[n_test:]
    train_texts, train_labels = zip(*train) if train else ([],[])
    test_texts, test_labels = zip(*test) if test else ([],[])
    return list(train_texts), list(test_texts), list(train_labels), list(test_labels)

def predict_batches(tokenizer, model, texts, batch_size=16, device=None):
    preds = []
    probs = []
    model.to(device or torch.device('cpu'))
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        if device:
            enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            soft = torch.nn.functional.softmax(logits, dim=-1)
            pred_ids = torch.argmax(soft, dim=-1).cpu().numpy().tolist()
            pred_probs = soft.cpu().numpy().tolist()
            preds.extend(pred_ids)
            probs.extend(pred_probs)
    return preds, probs

def main():
    texts, labels = load_data(CSV_PATH)
    if train_test_split:
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=1)
    else:
        train_texts, test_texts, train_labels, test_labels = simple_split(texts, labels, test_size=0.2, seed=1)

    # If not running in mock mode, prefer the trained checkpoint in results if available
    if not args.mock:
        # Search for the latest checkpoint in the results folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        
        best_checkpoint = None
        if os.path.exists(results_dir):
            checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by step number (e.g., checkpoint-500)
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                best_checkpoint = os.path.join(results_dir, checkpoints[-1])
                print(f"Found latest checkpoint: {best_checkpoint}")

        if best_checkpoint and os.path.exists(best_checkpoint):
            MODEL_DIR = best_checkpoint
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        else:
            # Fallback: try to use the lazy loader from predict.py (may point to saved model)
            try:
                import predict
                predict._load_model()
                tokenizer = predict.tokenizer
                model = predict.model
                if tokenizer is None or model is None:
                    raise RuntimeError("Model/tokenizer not loaded by predict._load_model()")
            except Exception:
                # Final fallback: load default path
                default_model = os.path.join(os.path.dirname(__file__), "..", "fake-news-bert-base-uncased")
                tokenizer = AutoTokenizer.from_pretrained(default_model)
                model = AutoModelForSequenceClassification.from_pretrained(default_model)

    # If mock flag is set, run a lightweight sklearn-based classifier to show expected output
    if args.mock:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except Exception:
            print("sklearn is required for mock mode. Install scikit-learn in your environment.")
            return

        vec = TfidfVectorizer(max_features=2000)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_test).tolist()
        # create probs as two-column soft-like scores
        try:
            probs_arr = clf.predict_proba(X_test)
            probs = probs_arr.tolist()
        except Exception:
            probs = [[0.5,0.5] for _ in preds]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preds, probs = predict_batches(tokenizer, model, test_texts, batch_size=16, device=device)

    # compute metrics
    y_true = list(test_labels)
    y_pred = preds

    if accuracy_score:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
        cm = confusion_matrix(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        if acc < 0.6:
            print(f"\n[WARNING] Accuracy is {acc:.2%}, which is close to random guessing (50%).")
            print("The model has not learned effectively. This is likely due to a very small training dataset.")
            print("Predictions from this model will be unreliable.\n")

        print(f"Precision (FAKE=1): {prec:.4f}")
        print(f"Recall (FAKE=1): {rec:.4f}")
        print(f"F1 (FAKE=1): {f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
    else:
        # simple accuracy & confusion if sklearn missing
        correct = sum(1 for a,b in zip(y_true,y_pred) if a==b)
        acc = correct / max(1,len(y_true))
        tp = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==1)
        tn = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==0)
        fp = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==1)
        fn = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==0)
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (FAKE=1): {prec:.4f}")
        print(f"Recall (FAKE=1): {rec:.4f}")
        print(f"F1 (FAKE=1): {f1:.4f}")
        print("Confusion Matrix (tp, fp, fn, tn):", tp, fp, fn, tn)

if __name__ == "__main__":
    main()
