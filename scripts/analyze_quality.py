"""
analyze_alignment_quality.py
Analyze claim–evidence alignment quality for hybrid retrieval output.

Metrics:
- Average similarity scores (dense, cross, BM25)
- Distribution histograms
- Example inspection: top-5 & bottom-5 matches
- Company-level summary
"""

import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# ========= CONFIG =========
INPUT_FILE = "data/processed/claim_evidence_pairs_hybrid_fixed.jsonl"
EXPORT_EXAMPLES = True
EXPORT_PATH = "data/processed/alignment_examples.csv"
SHOW_PLOTS = True
# ==========================


def load_pairs(path):
    print(f"📖 Loading pairs from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} pairs across {df['company'].nunique()} companies")
    return df


def basic_stats(df):
    print("\n📊 --- Basic Statistics ---")
    for col in ["bm25_score", "dense_score", "cross_score"]:
        if col in df.columns:
            print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}")

    # Number of unique claims per company
    comp_stats = df.groupby("company")["claim"].nunique().sort_values(ascending=False)
    print("\n🏢 Claims per company:")
    print(comp_stats.head(10))


def sample_examples(df):
    print("\n🔍 --- Example Pairs ---")

    # Sort by cross_score or hybrid proxy
    score_col = "cross_score" if "cross_score" in df.columns else "dense_score"
    df_sorted = df.sort_values(score_col, ascending=False)

    top_examples = df_sorted.head(5)
    bottom_examples = df_sorted.tail(5)

    print("\n✅ Top 5 matches:")
    for _, row in top_examples.iterrows():
        print(f"\nCompany: {row['company']}")
        print(f"Claim: {row['claim']}")
        print(f"Evidence: {row['evidence']}")
        print(f"{score_col}: {row[score_col]:.3f}")
        print("-" * 50)

    print("\n⚠️ Bottom 5 matches:")
    for _, row in bottom_examples.iterrows():
        print(f"\nCompany: {row['company']}")
        print(f"Claim: {row['claim']}")
        print(f"Evidence: {row['evidence']}")
        print(f"{score_col}: {row[score_col]:.3f}")
        print("-" * 50)

    if EXPORT_EXAMPLES:
        pd.concat([
            top_examples.assign(label="good"),
            bottom_examples.assign(label="bad")
        ]).to_csv(EXPORT_PATH, index=False)
        print(f"\n💾 Exported examples to {EXPORT_PATH}")


def visualize_distributions(df):
    score_cols = [c for c in ["bm25_score", "dense_score", "cross_score"] if c in df.columns]
    for col in score_cols:
        plt.figure()
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(False)
        plt.tight_layout()
        plt.show()


def main():
    df = load_pairs(INPUT_FILE)
    basic_stats(df)
    sample_examples(df)

    if SHOW_PLOTS:
        visualize_distributions(df)

    # Optional: simple quality heuristic
    high_conf = df[df["cross_score"] > 0.7] if "cross_score" in df.columns else df[df["dense_score"] > 0.6]
    low_conf = df[df["cross_score"] < 0.3] if "cross_score" in df.columns else df[df["dense_score"] < 0.4]
    print(f"\n📈 High-confidence matches: {len(high_conf)} ({len(high_conf)/len(df):.1%})")
    print(f"📉 Low-confidence matches: {len(low_conf)} ({len(low_conf)/len(df):.1%})")


if __name__ == "__main__":
    main()
