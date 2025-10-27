import pandas as pd
from newspaper import Article
from tqdm import tqdm
import time
import os
import json

# ===== CONFIG =====
INPUT_CSV = "data/nyt_evidence.csv"          # Input file containing NYT URLs
OUTPUT_CSV = "data/processed/nyt_articles.csv"      # Output CSV
OUTPUT_JSONL = "data/processed/nyt_articles.jsonl"  # Output JSONL
SAVE_EVERY = 10                      # Checkpoint every N articles
# ==================

def fetch_article(url, retries=3, delay=2):
    """
    Download and parse a NYT article.
    Returns (title, publish_date, text) or (None, None, None) on failure.
    """
    for attempt in range(retries):
        try:
            article = Article(url, language="en")
            article.download()
            article.parse()
            return article.title, article.publish_date, article.text
        except Exception as e:
            print(f"⚠️ Error fetching {url} (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    return None, None, None


def save_checkpoint(df, csv_path, jsonl_path):
    """Save progress to both CSV and JSONL formats."""
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.dropna(subset=["text"]).iterrows():
            record = {
                "url": row.get("url") or row.get("link"),
                "title": row.get("title"),
                "date": str(row.get("date")),
                "text": row.get("text")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"💾 Checkpoint saved: {len(df.dropna(subset=['text']))} articles")


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file '{INPUT_CSV}' not found.")

    df = pd.read_csv(INPUT_CSV)
    url_col = "url" if "url" in df.columns else "link"
    if url_col not in df.columns:
        raise ValueError("CSV must have a column named 'url' or 'link'.")

    # Add new columns if not present
    for col in ["title", "date", "text"]:
        if col not in df.columns:
            df[col] = None

    # Resume where you left off
    start_idx = df["text"].notna().sum()
    print(f"🔁 Resuming from index {start_idx} (already scraped {start_idx} articles)")

    for i in tqdm(range(start_idx, len(df)), desc="Scraping NYT articles"):
        url = df.loc[i, url_col]
        if pd.isna(url):
            continue

        title, date, text = fetch_article(url)
        df.loc[i, "title"] = title
        df.loc[i, "date"] = date
        df.loc[i, "text"] = text

        # Save progress every few articles
        if (i + 1) % SAVE_EVERY == 0:
            save_checkpoint(df, OUTPUT_CSV, OUTPUT_JSONL)

    # Final save
    save_checkpoint(df, OUTPUT_CSV, OUTPUT_JSONL)
    print(f"\n✅ Done! Saved {len(df.dropna(subset=['text']))} articles to '{OUTPUT_CSV}' and '{OUTPUT_JSONL}'")

if __name__ == "__main__":
    main()
