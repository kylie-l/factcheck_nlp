import json
from collections import defaultdict
import random
import csv

INPUT_PATH = "data/processed/paired_dataset_labeled_ollama.jsonl"
OUTPUT_PATH = "manual_review.csv"

# ---- Load dataset ----
data = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# ---- Bucket by label ----
buckets = defaultdict(list)
for item in data:
    label = item["label"]
    buckets[label].append(item)

print("Label counts:")
for label, items in buckets.items():
    print(f"{label}: {len(items)}")

# ---- Stratified random sample ----
sampled_sizes = {
    "SUPPORTS": 20,
    "REFUTES": 15,
    "NOT ENOUGH INFO": 20
}

sampled = {}
for label, k in sampled_sizes.items():
    sampled[label] = random.sample(buckets[label], k)

# ---- Flatten into CSV rows (one evidence sentence per row) ----
rows = []
for label, items in sampled.items():
    for item in items:
        claim = item["claim"]
        gold_label = item["label"]
        company = item.get("company", "")   # safely get company name

        for ev in item["evidence"]:
            evidence_sentence = ev["sentence"]

            rows.append([
                company,
                claim,
                evidence_sentence,
                gold_label,
                "",     # correct? (manual)
                "",     # reason
                ""      # notes
            ])

# ---- Write CSV ----
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["company", "claim", "evidence", "label", "correct", "reason", "notes"])
    writer.writerows(rows)

print(f"Saved manual review sheet → {OUTPUT_PATH}")