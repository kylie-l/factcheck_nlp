"""
guardian_extractor_filtered.py
Fetches Guardian articles that mention your companies and filters for environmental content.

Usage:
    python scripts/data_collection/guardian_extractor_filtered.py --pages 3
"""

import requests
import pandas as pd
from tqdm import tqdm
import time
import json
import os
import argparse

from dotenv import load_dotenv
from config.env_keywords import ENV_KEYWORDS

load_dotenv()

# ========= CONFIG =========
API_KEY = os.getenv("GUARDIAN_API_KEY")  # set with: export GUARDIAN_API_KEY="your_key"
BASE_URL = "https://content.guardianapis.com/search"
OUTPUT_CSV = "data/raw/guardian_articles.csv"
OUTPUT_JSONL = "data/raw/guardian_articles.jsonl"
PAGE_SIZE = 50
SAVE_EVERY = 3  # checkpoint every N companies
# ==========================


def fetch_guardian_articles(query, pages=1, delay=0.5):
    """Fetch Guardian articles mentioning the company name."""
    all_articles = []

    for page in range(1, pages + 1):
        params = {
            "api-key": API_KEY,
            "q": f'"{query}"',  # exact phrase search
            "page": page,
            "page-size": PAGE_SIZE,
            "show-fields": "headline,bodyText,byline,publication,webPublicationDate",
            "order-by": "relevance",
        }
        r = requests.get(BASE_URL, params=params)
        if r.status_code != 200:
            print(f"⚠️ Error {r.status_code} for query '{query}', page {page}")
            break

        data = r.json()
        results = data.get("response", {}).get("results", [])
        if not results:
            break

        for item in results:
            fields = item.get("fields", {})
            record = {
                "id": item.get("id"),
                "url": item.get("webUrl"),
                "title": fields.get("headline"),
                "date": fields.get("webPublicationDate"),
                "author": fields.get("byline"),
                "publication": fields.get("publication"),
                "text": fields.get("bodyText"),
                "section": item.get("sectionName"),
                "query": query,
            }
            all_articles.append(record)

        if len(results) < PAGE_SIZE:
            break
        time.sleep(delay)

    return all_articles


def filter_environmental(articles):
    """Keep only articles whose text contains environmental keywords."""
    filtered = []
    lower_keywords = [k.lower() for k in ENV_KEYWORDS]

    for art in articles:
        text = (art.get("text") or "").lower()
        if any(k in text for k in lower_keywords):
            filtered.append(art)
    return filtered


def save_checkpoint(records):
    """Save dataset to both CSV and JSONL."""
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(records)} unique filtered articles so far.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1, help="Pages per company query (50 per page)")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError("Missing GUARDIAN_API_KEY. Run: export GUARDIAN_API_KEY='your_key'")

    if not os.path.exists("companies.txt"):
        raise FileNotFoundError("Missing companies.txt (one company per line).")

    with open("companies.txt") as f:
        companies = [line.strip() for line in f if line.strip()]

    all_articles = {}
    company_count = 0

    for company in companies:
        company_count += 1
        print(f"\n🔍 [{company_count}/{len(companies)}] Fetching articles for: {company}")

        articles = fetch_guardian_articles(company, pages=args.pages)
        filtered = filter_environmental(articles)

        for art in filtered:
            if art["id"] not in all_articles:
                all_articles[art["id"]] = art

        if company_count % SAVE_EVERY == 0:
            save_checkpoint(list(all_articles.values()))

    save_checkpoint(list(all_articles.values()))
    print(f"\n✅ Done. Total unique environment-related articles: {len(all_articles)}")
    print(f"Files saved: {OUTPUT_CSV}, {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
