import json
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === CONFIG ===
DATA_PATH = "data/processed/paired_dataset_labeled_ollama.jsonl"  # your labeled dataset path
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]  # adjust if you only have 2 classes

# === LOAD DATA ===
true_labels = []
with open(DATA_PATH, "r") as f:
    for line in f:
        item = json.loads(line)
        true_labels.append(item["label"].strip().upper())

# === RANDOM BASELINE ===
random_preds = [random.choice(LABELS) for _ in true_labels]

from collections import Counter
print(Counter(true_labels))
print(Counter(random_preds))

# === EVALUATION ===
acc = accuracy_score(true_labels, random_preds)
f1_macro = f1_score(true_labels, random_preds, average="macro")
print(f"Random Baseline Accuracy: {acc:.3f}")
print(f"Random Baseline Macro-F1: {f1_macro:.3f}")
print("\nDetailed Report:")
print(classification_report(true_labels, random_preds, digits=3))
