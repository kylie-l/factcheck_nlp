"""
Implements SCIFACT-style two-stage retrieval for sustainability fact-checking:
1. Document-level retrieval (article-level relevance)
2. Sentence-level retrieval within top-k articles
3. Optional cross-encoder re-ranking for precise entailment scoring
"""

import pandas as pd
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from scipy.special import softmax
import numpy as np
import json
import os
import re
import torch

# ========= CONFIG =========
CORPORATE_CLAIMS = "data/processed/claims_environmental.jsonl"
GUARDIAN_ARTICLES = "data/raw/guardian_articles.jsonl"
OUTPUT_JSONL = "data/processed/claim_evidence_pairs_two_stage.jsonl"

# Retrieval configuration
TOP_K_DOCS = 5         # top-k articles to keep
TOP_K_SENTENCES = 5    # sentences per article
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ==========================


def load_corporate_claims(path):
    print(f"📖 Loading corporate claims from {path}")
    df = pd.read_json(path, lines=True)
    df = df.rename(columns={"claim_text": "claim"})
    return df


def load_guardian_articles(path):
    print(f"📖 Loading Guardian articles")
    articles = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(articles)
    df = df.dropna(subset=["text"])
    print(f"✅ Loaded {len(df)} articles")
    return df


def tokenize_preserve_case(text):
    """Simple tokenization that keeps capitalization."""
    return re.findall(r"\b[\w’']+\b", text)


def build_bm25_index(texts):
    tokenized = [tokenize_preserve_case(t) for t in texts]
    return BM25Okapi(tokenized), tokenized


def document_retrieval(claim_text, docs_df, bm25, dense_model, top_k=TOP_K_DOCS):
    """Retrieve the most relevant Guardian articles for a given claim."""
    # BM25
    bm25_scores = bm25.get_scores(tokenize_preserve_case(claim_text))
    bm25_idx = np.argsort(bm25_scores)[::-1][:50]  # prefilter
    bm25_candidates = docs_df.iloc[bm25_idx]

    # Dense
    claim_emb = dense_model.encode(claim_text, convert_to_tensor=True)
    doc_embs = dense_model.encode(bm25_candidates["text"].tolist(), convert_to_tensor=True)
    dense_scores = util.cos_sim(claim_emb, doc_embs)[0].cpu().numpy()

    bm25_candidates = bm25_candidates.assign(
        bm25_score=bm25_scores[bm25_idx],
        dense_score=dense_scores
    )
    bm25_candidates["doc_score"] = 0.5 * (
        bm25_candidates["dense_score"] +
        bm25_candidates["bm25_score"] / np.max(bm25_scores)
    )
    return bm25_candidates.sort_values("doc_score", ascending=False).head(top_k)


def sentence_retrieval(claim_text, article_row, dense_model, top_k=TOP_K_SENTENCES, context_window=1):
    """Retrieve top-k relevant sentences + surrounding context from one article."""
    text = article_row["text"]
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return []

    # Embed
    claim_emb = dense_model.encode(claim_text, convert_to_tensor=True)
    sent_embs = dense_model.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, sent_embs)[0].cpu().numpy()

    # Top-k indices
    ranked_idx = np.argsort(sims)[::-1][:top_k]
    context_idx = extract_with_context(sentences, ranked_idx, window=context_window)

    results = []
    for i in context_idx:
        results.append({
            "sentence": sentences[i],
            "dense_score": float(sims[i]),
            "article_title": article_row.get("title"),
            "url": article_row.get("url"),
            "date": article_row.get("date"),
        })
    return results



def rerank_with_crossencoder(claim_text, candidates, reranker):
    """Re-rank candidate sentences with an NLI model (entailment probability)."""
    pairs = [(claim_text, c["sentence"]) for c in candidates]
    logits = np.array(reranker.predict(pairs))
    probs = softmax(logits, axis=1)
    entailment_probs = probs[:, 2]
    labels = probs.argmax(axis=1)
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    for i, c in enumerate(candidates):
        c["cross_score"] = float(entailment_probs[i])
        c["cross_label"] = label_map[labels[i]]
    candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
    return candidates

def filter_docs_by_company(docs_df, company):
    """Keep only articles that mention the company."""
    pattern = re.compile(rf"\b{re.escape(company)}\b", re.IGNORECASE)
    filtered = docs_df[docs_df["text"].apply(lambda t: bool(pattern.search(t)))]
    return filtered if not filtered.empty else docs_df  # fallback if none


def extract_with_context(sentences, top_indices, window=1):
    """Include ±window sentences around each top match (deduplicated)."""
    context_indices = set()
    for i in top_indices:
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context_indices.update(range(start, end))
    return sorted(list(context_indices))

def main():
    nltk.download("punkt")

    claims_df = load_corporate_claims(CORPORATE_CLAIMS)
    docs_df = load_guardian_articles(GUARDIAN_ARTICLES)

    print(f"🔧 Loading dense model: {DENSE_MODEL}")
    dense_model = SentenceTransformer(DENSE_MODEL)
    reranker = CrossEncoder(CROSS_ENCODER_MODEL) if USE_CROSS_ENCODER else None

    bm25, _ = build_bm25_index(docs_df["text"].tolist())
    all_pairs = []

    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
        company, claim = row["company"], row["claim"]

        # Stage 1: document retrieval
        # Stage 1: document retrieval (then filter by company)
        top_docs = document_retrieval(claim, docs_df, bm25, dense_model)
        top_docs = filter_docs_by_company(top_docs, company)

        # Stage 2: sentence retrieval within top docs
        candidate_sentences = []
        for _, doc_row in top_docs.iterrows():
            sents = sentence_retrieval(claim, doc_row, dense_model)
            candidate_sentences.extend(sents)

        # Stage 3: re-rank with cross-encoder (optional)
        if USE_CROSS_ENCODER and candidate_sentences:
            candidate_sentences = rerank_with_crossencoder(claim, candidate_sentences, reranker)

        # Save pairs
        for ev in candidate_sentences:
            all_pairs.append({
                "company": company,
                "claim": claim,
                "filing": row.get("filing"),
                "source_file": row.get("source_file"),
                "evidence": ev["sentence"],
                "bm25_score": ev.get("bm25_score", 0),
                "dense_score": ev.get("dense_score", 0),
                "cross_score": ev.get("cross_score", 0),
                "cross_label": ev.get("cross_label", ""),
                "article_title": ev["article_title"],
                "url": ev["url"],
                "date": ev["date"]
            })

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(all_pairs)} claim–evidence pairs to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
