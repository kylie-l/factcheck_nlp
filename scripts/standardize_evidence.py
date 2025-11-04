"""
Standardize Guardian and GDELT evidence datasets and generate proper unique IDs.

Output file: data/processed/evidence_standardized.jsonl
Schema:
{
  "article_id": "guardian_000123",
  "source": "guardian",
  "url": "https://...",
  "title": "...",
  "companies": ["Microsoft", "Google"],
  "text": "Full article text ..."
}
"""

import json
import os
from tqdm import tqdm
import hashlib

# ====== CONFIG ======
GUARDIAN_PATH = "data/raw/guardian_articles.jsonl"
GDELT_PATH = "data/processed/gdelt_articles_with_text.jsonl"
OUTPUT_PATH = "data/processed/evidence_standardized.jsonl"

# ====== HELPERS ======
def load_jsonl(path):
    """Load JSONL safely (skip corrupt lines)."""
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def make_id(prefix, index, url=None):
    """Create a unique deterministic ID for each record."""
    if url:
        # stable hash from URL if present
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{prefix}_{h}"
    return f"{prefix}_{index:06d}"

def normalize_companies(val):
    """Ensure companies field is always a list of strings."""
    if not val:
        return []
    if isinstance(val, str):
        return [val.strip()]
    if isinstance(val, list):
        return [v.strip() for v in val if isinstance(v, str)]
    return []

# ====== LOAD ======
print("📖 Loading datasets...")
guardian_raw = load_jsonl(GUARDIAN_PATH)
gdelt_raw = load_jsonl(GDELT_PATH)
print(f"Loaded {len(guardian_raw)} Guardian | {len(gdelt_raw)} GDELT articles")

# ====== STANDARDIZE ======
standardized = []

# --- Guardian ---
print("\n📰 Processing Guardian...")
for i, art in enumerate(tqdm(guardian_raw)):
    url = art.get("url") or art.get("webUrl")
    text = art.get("text") or art.get("content") or art.get("body")
    # sometimes nested under "fields"
    if not text and isinstance(art.get("fields"), dict):
        text = art["fields"].get("bodyText")
    if not text or len(text.strip()) < 50:
        continue

    title = art.get("title") or art.get("headline") or art.get("webTitle")
    companies = normalize_companies(art.get("companies"))

    article_id = make_id("guardian", i, url)
    standardized.append({
        "article_id": article_id,
        "source": "guardian",
        "url": url,
        "title": title,
        "companies": companies,
        "text": text.strip()
    })

print(f"✅ Guardian standardized: {len(standardized)} entries")

# --- GDELT ---
print("\n🌎 Processing GDELT...")
for i, art in enumerate(tqdm(gdelt_raw)):
    url = art.get("url") or art.get("sourceUrl") or None
    text = art.get("content") or art.get("text") or art.get("body")
    if not text or len(text.strip()) < 50:
        continue

    title = art.get("title")
    companies = normalize_companies(art.get("companies") or art.get("company"))
    article_id = make_id("gdelt", i, url)

    standardized.append({
        "article_id": article_id,
        "source": "gdelt",
        "url": url,
        "title": title,
        "companies": companies,
        "text": text.strip()
    })

print(f"✅ GDELT standardized: {len(standardized)} total entries")

# ====== SAVE ======
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for entry in standardized:
        f.write(json.dumps(entry) + "\n")

print(f"\n💾 Saved standardized evidence dataset → {OUTPUT_PATH}")
print(f"Total combined articles: {len(standardized)}")
