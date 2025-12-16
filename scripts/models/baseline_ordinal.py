import json
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/gold_labeled.jsonl"
LABEL_KEYS = ["gold_label", "llm_score", "label"]
VALID_LABELS = {1, 2, 3, 4, 5}

# ============================================================
# LOAD DATA
# ============================================================

true_labels = []
relevance_scores = []
companies = []


def extract_label(item):
    """Return the first valid label (1–5) found in LABEL_KEYS."""
    for key in LABEL_KEYS:
        if key in item and item[key] is not None:
            try:
                val = int(item[key])
                if val in VALID_LABELS:
                    return val
            except:
                pass
    return None


with open(DATA_PATH, "r") as f:
    for line in f:
        item = json.loads(line)

        lbl = extract_label(item)
        if lbl is None:
            continue
        true_labels.append(lbl)

        if "company" in item and item["company"]:
            companies.append(item["company"].strip())

        if isinstance(item.get("evidence"), list):
            for ev in item["evidence"]:
                if "similarity" in ev:
                    relevance_scores.append(ev["similarity"])

true_labels = np.array(true_labels, dtype=int)
unique_labels = sorted(set(true_labels))

# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n=== Label Distribution (1–5) ===")
for l in unique_labels:
    print(f"Label {l}: {np.sum(true_labels == l)}")

# ============================================================
# ORDINAL METRIC HELPERS
# ============================================================

def compute_ordinal_metrics(preds, gold):
    preds = np.array(preds, dtype=float)
    gold = np.array(gold, dtype=float)
    abs_err = np.abs(preds - gold)
    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    within1 = np.mean(abs_err <= 1)
    return abs_err, mae, rmse, within1


def print_ordinal_metrics(name, preds, gold):
    abs_err, mae, rmse, within1 = compute_ordinal_metrics(preds, gold)

    print(f"\n=== Ordinal Metrics ({name}) ===")
    print(f"MAE:          {mae:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"Within-1 acc: {within1:.4f}")

    print("\nError distribution:")
    for d in range(0, 5):
        count = np.sum(abs_err == d)
        print(f" diff {d}: {count} ({count / len(abs_err):.3f})")

    print("\nPer-label MAE:")
    for l in unique_labels:
        idx = np.where(gold == l)[0]
        if len(idx) == 0:
            continue
        print(f" Label {l}: MAE={abs_err[idx].mean():.3f} (n={len(idx)})")


# ============================================================
# RANDOM BASELINE
# ============================================================

rng = np.random.default_rng()
random_preds = rng.choice(unique_labels, size=len(true_labels))

print("\n=== Random Baseline ===")
print("Prediction distribution:", Counter(random_preds))

print("\n--- Classification Metrics ---")
print(f"Accuracy:     {accuracy_score(true_labels, random_preds):.4f}")
print(f"Macro F1:     {f1_score(true_labels, random_preds, average='macro', zero_division=0):.4f}")
print(f"Micro F1:     {f1_score(true_labels, random_preds, average='micro'):.4f}")
print(f"Weighted F1:  {f1_score(true_labels, random_preds, average='weighted'):.4f}")

print("\nPer-class report:")
print(classification_report(true_labels, random_preds, digits=3, zero_division=0))

# Ordinal metrics (Random baseline)
print_ordinal_metrics("Random Baseline", random_preds, true_labels)

# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_class = Counter(true_labels).most_common(1)[0][0]
majority_preds = np.full_like(true_labels, majority_class)

print("\n=== Majority Baseline ===")
print(f"Majority label = {majority_class}")

print("\n--- Classification Metrics ---")
print(f"Accuracy:     {accuracy_score(true_labels, majority_preds):.4f}")
print(f"Macro F1:     {f1_score(true_labels, majority_preds, average='macro', zero_division=0):.4f}")
print(f"Micro F1:     {f1_score(true_labels, majority_preds, average='micro'):.4f}")
print(f"Weighted F1:  {f1_score(true_labels, majority_preds, average='weighted'):.4f}")

# Ordinal metrics (Majority baseline)
print_ordinal_metrics("Majority Baseline", majority_preds, true_labels)

# ============================================================
# RELEVANCE SCORE STATS
# ============================================================

if relevance_scores:
    arr = np.array(relevance_scores, dtype=float)
    print("\n=== Relevance Score Statistics ===")
    print(f"Count: {len(arr)}")
    print(f"Mean:  {arr.mean():.3f}")
    print(f"Std:   {arr.std():.3f}")
    print(f"Min:   {arr.min():.3f}")
    print(f"Max:   {arr.max():.3f}")
else:
    print("\nNo relevance scores found.")

# ============================================================
# COMPANY DISTRIBUTION
# ============================================================

if companies:
    c = Counter(companies)
    print("\n=== Company Distribution ===")
    print(f"Unique companies = {len(c)}")
    for company, count in c.most_common(10):
        print(f"  {company}: {count}")
else:
    print("\nNo company names found.")
