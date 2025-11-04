"""
gdelt_text_scraper.py
Fetches full article text for each GDELT article, filtering URLs, merging results,
and supporting resume-from-checkpoint.

Usage:
    python scripts/data_collection/gdelt_text_scraper.py --max 500
    python scripts/data_collection/gdelt_text_scraper.py        # full run
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from trafilatura import extract
from langdetect import detect

# ========= CONFIG =========
INPUT_JSONL = "data/raw/gdelt_articles.jsonl"
OUTPUT_JSONL = "data/processed/gdelt_articles_with_text.jsonl"
OUTPUT_CSV = "data/processed/gdelt_articles_with_text.csv"
LOGFILE = "data/processed/gdelt_failures.log"
WAIT_BETWEEN = 0.5
# ==========================


def load_unique_urls(input_path):
    """Load and deduplicate URLs from JSONL metadata."""
    records = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                url = data.get("url")
                if url and url.startswith("http") and url not in records:
                    records[url] = data
            except json.JSONDecodeError:
                continue
    print(f"✅ Loaded {len(records)} unique article URLs.")
    return records


def load_already_scraped(output_path):
    """Return set of URLs already processed (if checkpoint exists)."""
    scraped_urls = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    scraped_urls.add(data.get("url"))
                except json.JSONDecodeError:
                    continue
        print(f"🔁 Resuming from checkpoint: {len(scraped_urls)} articles already scraped.")
    return scraped_urls


def scrape_and_merge(records, scraped_urls, max_articles=None):
    """Scrape article text and merge back with metadata, skipping already done URLs."""
    scraped = []
    urls = [u for u in records.keys() if u not in scraped_urls]
    if max_articles:
        urls = urls[:max_articles]

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out, open(LOGFILE, "a", encoding="utf-8") as log:
        for url in tqdm(urls, desc="Scraping articles"):
            data = records[url]
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    log.write(f"{url}\tHTTP {r.status_code}\n")
                    continue

                text = extract(r.text)
                if not text or len(text.split()) < 50:
                    log.write(f"{url}\tToo short\n")
                    continue

                try:
                    if detect(text) != "en":
                        log.write(f"{url}\tNon-English\n")
                        continue
                except Exception:
                    pass

                data["text"] = text
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                scraped.append(data)

                time.sleep(WAIT_BETWEEN)

            except Exception as e:
                log.write(f"{url}\tError: {e}\n")
                continue

    print(f"✅ Successfully scraped {len(scraped)} new articles.")
    return scraped


def save_csv():
    """Convert JSONL to CSV for inspection."""
    rows = []
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved CSV → {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Limit number of URLs for testing")
    args = parser.parse_args()

    records = load_unique_urls(INPUT_JSONL)
    scraped_urls = load_already_scraped(OUTPUT_JSONL)
    scrape_and_merge(records, scraped_urls, max_articles=args.max)
    save_csv()


if __name__ == "__main__":
    main()
