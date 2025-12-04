import json
import random
import csv

INPUT_PATH = "data/processed/paired_dataset_cleaned_scaled.jsonl"
OUTPUT_PATH = "data/processed/annotation_sample.csv"

SAMPLE_SIZE = 80   # ← change to 80, 120, etc.
RANDOM_SEED = 42    # reproducibility

def main():
    # Load dataset
    items = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            items.append(json.loads(line))

    print(f"Loaded {len(items)} cleaned claim–evidence pairs")

    # Reproducible sampling
    random.seed(RANDOM_SEED)
    sample = random.sample(items, min(SAMPLE_SIZE, len(items)))

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "claim", "company", "evidence"])

        for item in sample:
            ev_sentences = [
                e.get("sentence", "").strip()
                for e in item.get("evidence", [])
                if e.get("sentence")
            ]
            ev = " || ".join(ev_sentences)

            writer.writerow([
                item.get("id", ""),
                item.get("claim", ""),
                item.get("company", ""),
                ev
            ])

    print(f"Saved {len(sample)} sampled items → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
