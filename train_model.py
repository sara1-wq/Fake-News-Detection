import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Configuration
MODEL_NAME = "distilbert-base-uncased" # Smaller model handles tiny datasets better
CSV_PATH = os.path.join(os.path.dirname(__file__), "news.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "fake-news-bert-base-uncased")
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10 # Increased to help the model find the signal in a tiny dataset

# 2. Load and Preprocess Data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Combine title and text for a richer input
    df['text'] = df['title'].fillna('') + " - " + df['text'].fillna('')
    # Ensure labels are integers
    df['label'] = df['label'].astype(int)
    return df

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Custom Trainer to handle class imbalance (Fake news bias)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Aggressive Weighting: Weigh the FAKE class (1) 5x more than the REAL class (0)
        # This forces the model to prioritize identifying FAKE news correctly.
        weights = torch.tensor([1.0, 5.0]).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 3. Main Training Logic
if __name__ == "__main__":
    # Load data
    df = load_data(CSV_PATH)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # FAKE (1) or REAL (0)
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-5, # Explicitly set a standard fine-tuning rate
        weight_decay=0.01,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch", # Fixed: keyword must be 'eval_strategy' in newer versions
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # Stop if overfitting
    )

    # Train the model
    print("--- Starting Model Training ---")
    trainer.train()
    print("--- Finished Model Training ---")

    # Save the final model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
