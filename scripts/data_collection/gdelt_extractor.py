"""
gdelt_extractor.py
Collects environment-related news about companies from the GDELT 2.0 Doc API.

Usage:
    python scripts/data_collection/gdelt_extractor.py --maxrecords 250 --lang English
"""

import os
import json
import time
import argparse
import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from config.env_keywords import ENV_KEYWORDS

# ========= CONFIG =========
OUTPUT_JSONL = "data/raw/gdelt_articles.jsonl"
OUTPUT_CSV = "data/raw/gdelt_articles.csv"
SAVE_EVERY = 2  # checkpoint every N companies
WAIT_BETWEEN = 1.0  # seconds between API calls
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# ==========================

load_dotenv()


def build_query(company, keywords):
    """Create a Boolean query combining company name and ENV keywords safely."""
    processed_keywords = []
    for k in keywords:
        k = k.strip()
        if " " in k:  # multi-word phrase → quote it
            processed_keywords.append(f'"{k}"')
        else:
            processed_keywords.append(k)
    kw_query = " OR ".join(processed_keywords)
    return f'"{company}" AND ({kw_query})'



import io
import csv

def fetch_articles(company, keywords, maxrecords=250, lang="English"):
    """Query the GDELT 2.0 Doc API directly with safe keyword chunking and fallback."""
    results = []
    CHUNK_SIZE = 5
    keyword_batches = [keywords[i:i + CHUNK_SIZE] for i in range(0, len(keywords), CHUNK_SIZE)]

    for batch in keyword_batches:
        query = build_query(company, batch)
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": maxrecords,
            "sourcelang": lang
        }

        for attempt in range(2):  # 1 retry if it fails
            try:
                import random

                MAX_RETRIES = 3

                for attempt in range(MAX_RETRIES):
                    try:
                        r = requests.get(BASE_URL, params=params, timeout=60)
                        if r.status_code == 200 and r.text.strip().startswith("{"):
                            data = r.json().get("articles", [])
                            print(f"✅ {len(data)} results for {company} batch {batch[:3]}")
                            for d in data:
                                results.append({
                                    "company": company,
                                    "query": query,
                                    "title": d.get("title"),
                                    "url": d.get("url"),
                                    "date": d.get("seendate"),
                                    "domain": d.get("domain"),
                                    "language": d.get("language"),
                                    "sourceCountry": d.get("sourceCountry"),
                                    "tone": d.get("tone"),
                                })
                            break  # success, exit retry loop

                        elif r.status_code == 200 and r.text.strip() == "":
                            print(f"ℹ️ No articles found for {company} batch {batch[:3]}")
                            break

                        else:
                            print(f"⚠️ Non-JSON or unexpected response for {company} batch {batch[:3]}")
                            print("Preview:", r.text[:100])
                            break

                    except requests.exceptions.ReadTimeout:
                        wait = (attempt + 1) * 10 + random.uniform(0, 5)
                        print(f"⏳ Timeout fetching {company} batch {batch[:3]} — retrying in {wait:.1f}s...")
                        time.sleep(wait)
                        continue

                    except Exception as e:
                        print(f"⚠️ Other error for {company} batch {batch[:3]}: {e}")
                        time.sleep(5)
                        continue

                if r.status_code != 200:
                    print(f"⚠️ HTTP {r.status_code} for {company} batch {batch[:3]}")
                    continue

                text = r.text.strip()

                # 1️⃣ Case: got valid JSON
                if text.startswith("{"):
                    data = r.json().get("articles", [])
                    for d in data:
                        results.append({
                            "company": company,
                            "query": query,
                            "title": d.get("title"),
                            "url": d.get("url"),
                            "date": d.get("seendate"),
                            "domain": d.get("domain"),
                            "language": d.get("language"),
                            "sourceCountry": d.get("sourceCountry"),
                            "tone": d.get("tone"),
                        })
                    break  # exit retry loop if successful

                # 2️⃣ Case: fallback to CSV mode if non-JSON
                else:
                    params["format"] = "csv"
                    r_csv = requests.get(BASE_URL, params=params, timeout=30)
                    csv_text = r_csv.text.strip()
                    if csv_text.startswith("URL"):
                        reader = csv.DictReader(io.StringIO(csv_text))
                        for row in reader:
                            results.append({
                                "company": company,
                                "query": query,
                                "title": row.get("TITLE"),
                                "url": row.get("URL"),
                                "date": row.get("DATE"),
                                "domain": row.get("DOMAIN"),
                                "language": row.get("LANGUAGE"),
                                "sourceCountry": row.get("SOURCECOUNTRY"),
                                "tone": row.get("TONE"),
                            })
                        break
                    else:
                        print(f"⚠️ Non-JSON and non-CSV response for {company} batch {batch[:3]}")
                        print("Preview:", text[:120])
                        break

            except Exception as e:
                print(f"⚠️ Exception fetching {company} batch {batch[:3]}: {e}")
                time.sleep(2)
        time.sleep(1.0)

    return results


def save_checkpoint(records):
    """Save dataset to JSONL and CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(records)} GDELT articles so far.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxrecords", type=int, default=250, help="Max records per query (≤250)")
    parser.add_argument("--lang", type=str, default="English", help="Source language filter")
    args = parser.parse_args()

    if not os.path.exists("companies.txt"):
        raise FileNotFoundError("Missing companies.txt (one company per line).")

    with open("companies.txt") as f:
        companies = [line.strip() for line in f if line.strip()]

    all_articles = {}
    company_count = 0

    for company in companies:
        company_count += 1
        print(f"\n🔍 [{company_count}/{len(companies)}] Fetching GDELT results for: {company}")

        articles = fetch_articles(company, ENV_KEYWORDS, maxrecords=args.maxrecords, lang=args.lang)

        for art in articles:
            url = art.get("url")
            if not url:
                continue
            if url not in all_articles:
                all_articles[url] = art

        if company_count % SAVE_EVERY == 0:
            save_checkpoint(list(all_articles.values()))

        time.sleep(WAIT_BETWEEN)

    save_checkpoint(list(all_articles.values()))
    print(f"\n✅ Done. Total unique GDELT articles: {len(all_articles)}")
    print(f"Files saved: {OUTPUT_CSV}, {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
