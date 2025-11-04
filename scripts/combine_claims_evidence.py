"""
Create a FEVER-style dataset using the standardized evidence file.
Each output includes the actual evidence sentences (text).

Inputs:
- claims_full.jsonl: [{"id": ..., "claim": "...", "company": "..."}]
- evidence_standardized.jsonl: [
    {
      "article_id": "guardian_000123",
      "source": "guardian",
      "url": "...",
      "companies": ["Microsoft"],
      "text": "... full article text ..."
    }
  ]

Output:
- paired_dataset_with_text.jsonl
  [
    {
      "id": "claim_001",
      "label": "SUPPORTS",
      "claim": "...",
      "company": "Microsoft",
      "evidence": [
        {
          "article_id": "guardian_000123",
          "source": "guardian",
          "url": "...",
          "sent_idx": 3,
          "sentence": "Microsoft said on Thursday that it will reach net zero emissions by 2030.",
          "similarity": 0.82
        }
      ]
    }
  ]
"""

import json
import re
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download("punkt")

# ====== CONFIG ======
CLAIMS_PATH = "data/processed/claims_full.jsonl"
EVIDENCE_PATH = "data/processed/evidence_standardized.jsonl"
OUTPUT_PATH = "data/processed/paired_dataset_with_text.jsonl"

TOP_K_SENTENCES = 3
SIM_THRESHOLD = 0.35
WINDOW_SIZE = 1

# ====== HELPERS ======
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def build_company_regex(company):
    """Make a loose, case-insensitive regex for company mentions."""
    core = re.escape(company)
    return re.compile(rf"\b{core}(?:\s*,?\s*(?:Inc|Corp|Corporation|PLC|Ltd|LLC|SE))?\b", re.IGNORECASE)

def sent_tokenize_filtered(text):
    """Sentence-split and filter overly short or long sentences."""
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if 20 < len(s) < 400]

# ====== LOAD DATA ======
print("📖 Loading data...")
claims = load_jsonl(CLAIMS_PATH)
evidence_articles = load_jsonl(EVIDENCE_PATH)
print(f"Loaded {len(claims)} claims | {len(evidence_articles)} standardized evidence articles")

# ====== MODEL ======
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ====== MATCHING ======
fever_data = []
no_company, no_match = 0, 0

print("🔍 Matching claims to evidence (from standardized file)...")

for claim in tqdm(claims, desc="Processing claims"):
    claim_text = claim.get("claim") or claim.get("claim_text")
    company = claim.get("company") or claim.get("Company")
    if not claim_text or not company:
        no_company += 1
        continue

    claim_text = claim_text.strip()
    company = company.strip()
    comp_re = build_company_regex(company)
    claim_id = claim.get("id") or claim.get("claim_id")

    # 1️⃣ Filter articles mentioning this company in metadata
    relevant_articles = [
        art for art in evidence_articles
        if any(company.lower() == c.lower() for c in art.get("companies", []))
    ]

    # 2️⃣ Fallback: search in text if not found in metadata
    if not relevant_articles:
        relevant_articles = [art for art in evidence_articles if comp_re.search(art["text"])]

    if not relevant_articles:
        no_match += 1
        continue

    # 3️⃣ Extract candidate sentences near company mentions
    candidate_sents, meta = [], []
    for art in relevant_articles:
        sentences = nltk.sent_tokenize(art["text"])
        hits = [i for i, s in enumerate(sentences) if comp_re.search(s)]
        for i in hits:
            start, end = max(0, i - WINDOW_SIZE), min(len(sentences), i + WINDOW_SIZE + 1)
            for j in range(start, end):
                s = sentences[j].strip()
                if 20 < len(s) < 400:
                    candidate_sents.append((s, art, j))

    if not candidate_sents:
        no_match += 1
        continue

    # 4️⃣ Compute semantic similarity
    with torch.no_grad():
        sent_texts = [s[0] for s in candidate_sents]
        sent_embs = model.encode(sent_texts, convert_to_tensor=True, show_progress_bar=False)
        claim_emb = model.encode(claim_text, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(claim_emb, sent_embs)[0]
        k = min(TOP_K_SENTENCES, len(sims))
        topk = sims.topk(k)

    evidence_entries = []
    for idx, score in zip(topk.indices, topk.values):
        if float(score) < SIM_THRESHOLD:
            continue
        sent, art, sent_idx = candidate_sents[int(idx)]
        evidence_entries.append({
            "article_id": art["article_id"],
            "source": art["source"],
            "url": art.get("url"),
            "sent_idx": sent_idx,
            "sentence": sent,
            "similarity": float(score)
        })

    if evidence_entries:
        fever_data.append({
            "id": claim_id,
            "label": "SUPPORTS",
            "claim": claim_text,
            "company": company,
            "evidence": evidence_entries
        })

print(f"✅ Matched {len(fever_data)} claim–evidence pairs.")
print(f"ℹ️ Skipped {no_company} (no company) and {no_match} (no matches).")

# ====== SAVE ======
with open(OUTPUT_PATH, "w") as f:
    for entry in fever_data:
        f.write(json.dumps(entry) + "\n")

print(f"💾 Saved dataset with actual evidence text → {OUTPUT_PATH}")
