import json
import csv

FULL_DATASET = "data/processed/paired_dataset_cleaned_scaled.jsonl"
ANNOTATED = "data/processed/annotated_sample_labeled.csv"

OUTPUT_UNLABELED = "data/processed/silver_unlabeled.jsonl"
OUTPUT_GOLD = "data/processed/gold_labeled.jsonl"


def load_annotations(csv_path):
    """
    Load annotated rows and return:
      - gold_ids: set of annotated IDs
      - gold_map: mapping id -> {gold_label}
    Only 'id' and 'label' are required.
    """
    gold_ids = set()
    gold_map = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cid = row.get("id", "").strip()
            if not cid:
                continue

            # label column may be named 'label' or 'score'
            label_raw = (
                row.get("label", "").strip()
                or row.get("score", "").strip()
            )

            try:
                label_val = int(label_raw)
            except:
                label_val = None

            gold_ids.add(cid)
            gold_map[cid] = {
                "gold_label": label_val
            }

    return gold_ids, gold_map


def main():
    print("Loading annotations...")
    gold_ids, gold_map = load_annotations(ANNOTATED)
    print(f"✓ Loaded {len(gold_ids)} annotated IDs")

    unlabeled = []
    gold_items = []

    print("Processing full dataset...")
    with open(FULL_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = item.get("id", "").strip()

            if cid in gold_ids:
                merged = {**item, **gold_map[cid]}
                gold_items.append(merged)
            else:
                unlabeled.append(item)

    print(f"Silver (unlabeled): {len(unlabeled)}")
    print(f"Gold (labeled): {len(gold_items)}")

    # Save unlabeled silver dataset
    with open(OUTPUT_UNLABELED, "w", encoding="utf-8") as f:
        for item in unlabeled:
            f.write(json.dumps(item) + "\n")

    # Save gold labeled dataset
    with open(OUTPUT_GOLD, "w", encoding="utf-8") as f:
        for item in gold_items:
            f.write(json.dumps(item) + "\n")

    print("\n✓ Done!")
    print(f"  Silver unlabeled → {OUTPUT_UNLABELED}")
    print(f"  Gold labeled →     {OUTPUT_GOLD}")


if __name__ == "__main__":
    main()
