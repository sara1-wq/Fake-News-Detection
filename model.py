import os
# Set environment variable to fix WinError 1114 (DLL initialization failed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# DLL PRE-LOADER (Fix for WinError 1114 on Python 3.14)
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            torch_lib_path = os.path.join(os.path.dirname(spec.origin), "lib")
            c10_dll = os.path.join(torch_lib_path, "c10.dll")
            if os.path.exists(c10_dll):
                ctypes.CDLL(c10_dll)
    except Exception:
        pass

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Import torch after numpy to fix potential DLL initialization errors (WinError 1114)
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

set_seed(1)

# the model we gonna train, base uncased BERT
model_name = "bert-base-uncased"
# max sequence length for each document/sentence 
max_length = 512
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

# Load the news data from CSV
news_csv_path = "news.csv"
if not os.path.exists(news_csv_path):
    news_csv_path = os.path.join(os.path.dirname(__file__), "news.csv")
news_d = pd.read_csv(news_csv_path)

# Normalize labels if they are strings (REAL/FAKE -> 0/1)
if news_d['label'].dtype == object:
    # Robust normalization: strip whitespace, uppercase
    news_d['label'] = news_d['label'].astype(str).str.strip().str.upper()
    label_map = {'FAKE': 1, 'REAL': 0}
    # Filter to only keep known labels
    news_d = news_d[news_d['label'].isin(label_map.keys())]
    news_d['label'] = news_d['label'].map(label_map)

# Drop records with unknown labels instead of defaulting to 0
news_d = news_d.dropna(subset=['label'])
news_d['label'] = news_d['label'].astype(int)

print(f"Label distribution: {news_d['label'].value_counts().to_dict()} (0=REAL, 1=FAKE)")

# Fill missing text/title/author instead of dropping rows
news_d['text'] = news_d['text'].fillna("")
if "title" in news_d.columns:
    news_d['title'] = news_d['title'].fillna("")
if "author" in news_d.columns:
    news_d['author'] = news_d['author'].fillna("")

news_df = news_d[news_d['text'].str.strip().astype(bool)]

def prepare_data(df, test_size=0.2, include_title=True, include_author=True, oversample_minority=True):
    texts = []
    labels = []
    for i in range(len(df)):
        text = str(df["text"].iloc[i])
        label = int(df["label"].iloc[i])
        
        prefix = ""
        if include_title and "title" in df.columns:
            title_val = str(df["title"].iloc[i])
            if title_val.strip():
                prefix += title_val + " - "
        if include_author and "author" in df.columns:
            author_val = str(df["author"].iloc[i])
            if author_val.strip():
                prefix += author_val + " : "
        
        full_text = prefix + text
        
        if full_text.strip():
            texts.append(full_text)
            labels.append(label)

    if len(set(labels)) < 2:
        print(f"WARNING: Training data only contains labels {set(labels)}. Model will be biased!")

    # Optionally oversample the minority class to reduce bias towards majority
    if oversample_minority:
        counter = Counter(labels)
        if len(counter) == 2:
            majority_label = 0 if counter[0] >= counter[1] else 1
            minority_label = 1 - majority_label
            maj_count = counter[majority_label]
            min_count = counter[minority_label]
            if min_count > 0 and maj_count > min_count:
                minority_samples = [t for t, l in zip(texts, labels) if l == minority_label]
                import random
                # duplicate random minority samples until balanced roughly
                while len([l for l in labels if l == minority_label]) < maj_count:
                    sample = random.choice(minority_samples)
                    texts.append(sample)
                    labels.append(minority_label)

    # stratify the split to preserve label distribution between train/validation
    try:
        return train_test_split(texts, labels, test_size=test_size, stratify=labels)
    except Exception:
        # fallback to non-stratified split if stratify fails (e.g., tiny classes)
        return train_test_split(texts, labels, test_size=test_size)

train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)

# print(len(train_texts), len(train_labels))
# print(len(valid_texts), len(valid_labels))

# shuffle training data deterministically before tokenization
train_data = list(zip(train_texts, train_labels))
random.shuffle(train_data)
train_texts, train_labels = zip(*train_data)
train_texts, train_labels = list(train_texts), list(train_labels)

# tokenize the dataset, truncate when passed `max_length`, pad if shorter than `max_length`
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # store labels as scalar tensors (not length-1 lists)
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

# load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# compute class weights from training labels to mitigate imbalance
try:
    from collections import Counter
    counter = Counter(train_labels)
    total = sum(counter.values()) if len(counter) > 0 else 1
    # aggressive re-weighting: amplify minority class weight so trainer strongly
    # penalizes misclassification of the minority (likely FAKE)
    class_weights = [1.0, 1.0]
    if len(counter) == 2:
        maj_label = 0 if counter[0] >= counter[1] else 1
        min_label = 1 - maj_label
        maj_count = counter[maj_label]
        min_count = counter[min_label] if counter[min_label] > 0 else 1
        ratio = float(maj_count) / float(min_count)
        # amplify the minority weight by the ratio times a small multiplier
        amp = max(2.0, ratio * 1.5)
        class_weights = [1.0, 1.0]
        class_weights[min_label] = float(amp)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
except Exception:
    class_weights = None

# define training arguments
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=8,              # more epochs for stronger learning
        per_device_train_batch_size=8,   # batch size per device during training
        gradient_accumulation_steps=2,   # simulate larger batch without OOM
        per_device_eval_batch_size=32,   # larger eval batch if memory allows
        learning_rate=2e-05,             # slightly lower learning rate for finetuning
        weight_decay=0.01,
        warmup_steps=200,                # increased warmup steps
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training
        metric_for_best_model='accuracy',
        logging_steps=50,                # more frequent logging
        save_strategy="epoch",
        eval_strategy="epoch",         # evaluate at end of each epoch
        save_total_limit=3,
        seed=1,
        fp16=False,
    )

    # Use a Trainer subclass that applies class-weighted cross-entropy if available
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # ensure labels are 1D
            if labels is not None:
                labels = labels.view(-1)
            outputs = model(**inputs)
            logits = outputs.logits
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.num_labels), labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    # train the model
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()

    # saving the fine tuned model & tokenizer
    model_path = "fake-news-bert-base-uncased"
    try:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    except Exception as e:
        print(f"Warning: model.save_pretrained failed: {e}\nFalling back to torch.save of state_dict.")
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
        # save config and tokenizer so model can be reloaded
        model.config.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    