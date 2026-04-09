from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

MODEL_BASE  = "google/muril-base-cased"
OUTPUT_DIR  = "./muril-sentimix"
NUM_LABELS  = 3

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ── Load dataset ──────────────────────────────────────────────
raw = load_dataset("AmaanP314/youtube-comment-sentiment")["train"]

# Rename and encode columns
df = raw.to_pandas()[["CommentText", "Sentiment"]].dropna()
df = df.rename(columns={"CommentText": "text"})
df["labels"] = df["Sentiment"].str.lower().map(LABEL2ID)
df = df.dropna(subset=["labels"])
df["labels"] = df["labels"].astype(int)

# Sample 100k to keep training fast (remove cap if you have GPU + time)
df = df.sample(n=min(100_000, len(df)), random_state=42).reset_index(drop=True)

split = Dataset.from_pandas(df[["text", "labels"]]).train_test_split(test_size=0.1, seed=42)

# ── Tokenizer ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

split = split.map(tokenize, batched=True)
split.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
)

# ── Metrics ───────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# ── Training args ─────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)

# ── Train ─────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
