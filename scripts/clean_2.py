import json
import re
from tqdm import tqdm

# ================================
# Paths
# ================================
INPUT_PATH = "data/processed/paired_dataset_cleaned.jsonl"
OUTPUT_PATH = "data/processed/paired_dataset_cleaned_scaled.jsonl"
REMOVED_PATH = "data/processed/removed_claims_scaled.jsonl"  # (optional) to inspect removed ones

# ================================
# Heading Detector
# ================================
def looks_like_heading(claim: str) -> bool:
    """
    Removes:
    - generic headings ("Appendix", "Summary", etc.)
    - slide/page headings ("38 Resource Efficiency ...")
    - multi-topic noun lists
    - claims with no verbs
    - table-of-contents style lines
    - overly long or overly short strings
    - high capitalized-word ratio (slides)
    """
    text = claim.strip()
    lower = text.lower()
    words = text.split()

    # Detect many Title-Case tokens at the start (table/report headings)
    tokens = text.split()
    capitalized_runs = 0

    for t in tokens[:12]:  # check first 10–12 words
        if t[:1].isupper():
            capitalized_runs += 1

    if capitalized_runs >= 8:
        return True

    # 1. Very short strings are headings ("Appendix", "Summary")
    if len(words) <= 3:
        return True

    # 3. Slide/page number headings (e.g., "38 Resource Efficiency")
    if re.match(r"^\s*\d{1,3}\s+[A-Za-z]", text):
        return True

    # 4. NEW: Excessive digits at the start (chart axes)
    first_30 = text[:30]
    if sum(c.isdigit() for c in first_30) >= 8:
        return True

    # Detect sequences of 3+ numbers separated by whitespace
    if re.search(r"(?:\b\d+(?:\.\d+)?%?\b[\s,]+){3,}\b\d+(?:\.\d+)?%?\b", text):
        return True

    # 5. No punctuation AND many words → outline/list
    if not any(ch in text for ch in ".,:;!?"):
        if len(words) > 6:
            return True

    # 6. Very high capitalization → slide title mashup
    cap_ratio = sum(w[:1].isupper() for w in words) / len(words)
    if cap_ratio > 0.5:
        return True

    # NEW rule: remove anything containing a URL
    if "http://" in lower or "https://" in lower or "www." in lower:
        return True

    UI_JUNK = [
        "more information", "view", "recommended action", 
        "click here", "download", "learn more", "profile"
    ]

    if any(phrase in lower for phrase in UI_JUNK):
        return True

    # Remove alternating label-number sequences (table rows)
    if re.search(r"[A-Za-z]+\s+\d+(\.\d+)?\s+[A-Za-z]+\s+\d+(\.\d+)?", text):
        return True

    return False

def assign_ids(items):
    for i, item in enumerate(items):
        item["id"] = f"claim_{i+1:05d}"
    return items

# ================================
# Cleaning Script
# ================================
def main():
    cleaned = 0
    removed = 0

    # Load dataset
    with open(INPUT_PATH, "r") as f:
        items = [json.loads(line) for line in f]

    output = []
    removed_items = []

    for item in tqdm(items, desc="Filtering claims"):
        claim = item.get("claim", "").strip()

        # Remove heading-like entries
        if looks_like_heading(claim):
            removed += 1
            removed_items.append(item)
            continue

        # Remove the `label` field if present
        item.pop("label", None)

        output.append(item)
        cleaned += 1
    
    output = assign_ids(output)

    # Save cleaned dataset
    with open(OUTPUT_PATH, "w") as out:
        for item in output:
            out.write(json.dumps(item) + "\n")

    # Save removed claims for inspection
    with open(REMOVED_PATH, "w") as out:
        for item in removed_items:
            out.write(json.dumps(item) + "\n")

    print("\n🎉 Cleaning complete!")
    print(f"✔ Kept: {cleaned} valid claim–evidence pairs")
    print(f"✘ Removed: {removed} heading-like claims")
    print(f"💾 Cleaned file saved to: {OUTPUT_PATH}")
    print(f"🗂 Removed items saved to: {REMOVED_PATH}")


if __name__ == "__main__":
    main()
