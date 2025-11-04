import json

INPUT_FILE = "data/processed/paired_dataset_with_text.jsonl"
OUTPUT_FILE = "data/processed/paired_dataset_cleaned.jsonl"

def clean_jsonl(input_path, output_path):
    cleaned = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            # --- Deduplicate evidence per claim ---
            seen_sentences = set()
            unique_evidence = []

            for ev in item.get("evidence", []):
                sent = ev.get("sentence", "")
                if sent and sent not in seen_sentences:
                    seen_sentences.add(sent)
                    unique_evidence.append(ev)

            item["evidence"] = unique_evidence
            cleaned.append(item)

    # --- Write output with real Unicode characters ---
    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in cleaned:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Cleaned {len(cleaned)} claims written to {output_path}")

if __name__ == "__main__":
    clean_jsonl(INPUT_FILE, OUTPUT_FILE)