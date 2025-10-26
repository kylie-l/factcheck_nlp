import requests
import pandas as pd
import time
from tqdm import tqdm
from config.env_keywords import ENV_KEYWORDS as KEYWORDS
from dotenv import load_dotenv
import os
import random

load_dotenv()

COMPANIES_FILE = "companies.txt"
OUTPUT_FILE = "~/data/nyt_evidence.csv"

# =====================
# CONFIG
# =====================
NYT_API_KEY = os.getenv("NYTAPI_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Read company names from file (one per line)
def load_companies(path=COMPANIES_FILE):
    with open(path, "r") as f:
        companies = [line.strip() for line in f if line.strip()]
    return companies
    

# =====================
# HELPERS
# =====================

def get_nyt_articles(query, start_date="2023-01-01", end_date="2025-01-01", max_pages=3):
    """Fetch up to max_pages of NYT articles for a given query, with backoff."""
    all_results = []

    for page in range(max_pages):
        params = {
            "q": query,
            "api-key": NYT_API_KEY,
            "begin_date": start_date.replace("-", ""),
            "end_date": end_date.replace("-", ""),
            "page": page,
            "sort": "relevance",
        }

        # Retry with exponential backoff
        for attempt in range(5):
            r = requests.get(BASE_URL, params=params)
            if r.status_code == 200:
                break
            elif r.status_code == 429:
                wait = 60 + random.randint(15, 45)
                print(f"Rate limit hit (429). Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"Error {r.status_code}: {r.text[:120]}")
                return all_results  # stop paging on hard error

        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            break

        for doc in docs:
            all_results.append({
                "source": "NYT",
                "headline": doc["headline"].get("main", ""),
                "pub_date": doc.get("pub_date", ""),
                "url": doc.get("web_url", ""),
                "section": doc.get("section_name", ""),
                "snippet": doc.get("snippet", "") or doc.get("lead_paragraph", ""),
                "query": query
            })

        time.sleep(12)  # stay under 10 requests/min

    return all_results


def save_progress(df, out_path=OUTPUT_FILE):
    """Append new rows to the output file safely, removing duplicates."""
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        before = len(existing)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=["url"], inplace=True)
        added = len(combined) - before
        combined.to_csv(out_path, index=False)
        print(f"🧹 Deduplicated and saved: {added} new rows added.")
    else:
        df.drop_duplicates(subset=["url"], inplace=True)
        df.to_csv(out_path, index=False)
        print(f"✅ Created new file with {len(df)} rows.")



def load_existing_queries(out_path=OUTPUT_FILE):
    """Return a set of (company, keyword) pairs already saved."""
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path, usecols=["company", "keyword"]).drop_duplicates()
    return set(zip(df.company, df.keyword))


# =====================
# MAIN LOGIC
# =====================
def collect_nyt_articles():
    companies = load_companies(COMPANIES_FILE)
    print(f"Loaded {len(companies)} companies from {COMPANIES_FILE}")

    completed = load_existing_queries(OUTPUT_FILE)
    print(f"Resuming... {len(completed)} (company, keyword) pairs already processed.\n")

    for company in tqdm(companies, desc="Collecting NYT articles"):
        for kw in KEYWORDS:
            if (company, kw) in completed:
                continue  # skip previously done pairs

            query = f"{company} {kw}"
            articles = get_nyt_articles(query)

            if not articles:
                continue

            df = pd.DataFrame(articles)
            df["company"] = company
            df["keyword"] = kw

            save_progress(df)
            print(f"Saved {len(df)} new rows for {company} – {kw}")

    print("✅ All done! Results saved to nyt_evidence.csv")


# =====================
# MAIN ENTRY
# =====================
if __name__ == "__main__":
    collect_nyt_articles()