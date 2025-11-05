import json
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import numpy as np

# === CONFIG ===
DATA_PATH = "data/processed/paired_dataset_labeled_ollama.jsonl"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# === LOAD DATA ===
true_labels = []
relevance_scores = []
companies = []

with open(DATA_PATH, "r") as f:
    for line in f:
        item = json.loads(line)
        label = item["label"].strip().upper()
        true_labels.append(label)

        # Collect company name if available
        if "company" in item and item["company"]:
            companies.append(item["company"].strip())

        # Collect all similarity scores from evidence
        if "evidence" in item and isinstance(item["evidence"], list):
            for ev in item["evidence"]:
                if "similarity" in ev:
                    relevance_scores.append(ev["similarity"])

# === LABEL DISTRIBUTION ===
print("=== Label Distribution ===")
print(Counter(true_labels))

# === RANDOM BASELINE ===
random_preds = [random.choice(LABELS) for _ in true_labels]
print("\n=== Random Prediction Distribution ===")
print(Counter(random_preds))

# === EVALUATION ===
acc = accuracy_score(true_labels, random_preds)
f1_macro = f1_score(true_labels, random_preds, average="macro", zero_division=0)
print(f"\nRandom Baseline Accuracy: {acc:.3f}")
print(f"Random Baseline Macro-F1: {f1_macro:.3f}")
print("\nDetailed Report:")
print(classification_report(true_labels, random_preds, digits=3, zero_division=0))

# === MAJORITY BASELINE ===
majority = Counter(true_labels).most_common(1)[0][0]
maj_preds = [majority] * len(true_labels)
acc_majority = accuracy_score(true_labels, maj_preds)
f1_majority = f1_score(true_labels, maj_preds, average="macro", zero_division=0)
print(f"\nMajority Baseline Accuracy: {acc_majority:.3f}")
print(f"Majority Baseline Macro-F1: {f1_majority:.3f}")

# === RELEVANCE SCORE STATS ===
if relevance_scores:
    mean_rel = np.mean(relevance_scores)
    std_rel = np.std(relevance_scores)
    min_rel = np.min(relevance_scores)
    max_rel = np.max(relevance_scores)

    print("\n=== Relevance Score Statistics ===")
    print(f"Total evidence sentences: {len(relevance_scores)}")
    print(f"Mean similarity: {mean_rel:.3f}")
    print(f"Std deviation: {std_rel:.3f}")
    print(f"Min similarity: {min_rel:.3f}")
    print(f"Max similarity: {max_rel:.3f}")
else:
    print("\nNo relevance scores found in dataset.")

# === COMPANY DISTRIBUTION ===
if companies:
    company_counts = Counter(companies)
    total_companies = len(set(companies))
    print("\n=== Company Distribution ===")
    print(f"Total unique companies: {total_companies}")
    print("Top 10 companies:")
    for company, count in company_counts.most_common(10):
        print(f"  {company}: {count}")
else:
    print("\nNo company information found in dataset.")
