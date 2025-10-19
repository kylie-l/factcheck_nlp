import os
import json
import requests
import time
from tqdm import tqdm

HEADERS = {"User-Agent": "Kylie Liang (University of Michigan, fact-checking project; liangk@umich.edu)", "Accept-Encoding": "gzip, deflate"}

# Base URLs
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"

def load_companies(path="companies.txt"):
    """Read company names from text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def load_ticker_map():
    """Load the SEC’s official ticker → CIK mapping."""
    r = requests.get(TICKER_URL, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    return {v["title"].lower(): v["cik_str"] for v in data.values()}

def find_best_match(company_name, ticker_map):
    """Find the closest match for company name in the SEC mapping."""
    name = company_name.lower()
    for title, cik in ticker_map.items():
        if name in title:  # simple substring match
            return cik, title
    return None, None

def get_recent_filings(cik, form_type="10-K", limit=3):
    """Return a list of recent filings (HTML URLs) for a given company CIK."""
    url = SUBMISSIONS_URL.format(cik=cik)
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    results = []
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form == form_type:
            accession = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            results.append(filing_url)
            if len(results) >= limit:
                break
    return results

def download_filing(url, save_path):
    """Download and save a filing HTML page."""
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(r.text)

if __name__ == "__main__":
    companies = load_companies()
    ticker_map = load_ticker_map()
    os.makedirs("data/raw", exist_ok=True)

    for company in companies:
        cik, matched_name = find_best_match(company, ticker_map)
        if not cik:
            print(f"⚠️  No CIK found for {company}")
            continue

        print(f"\n=== {matched_name} ({cik}) ===")
        try:
            filings = get_recent_filings(cik)
            for url in tqdm(filings, desc=f"Downloading {company}"):
                fname = url.split("/")[-1].replace(".htm", f"_{company}.html")
                save_path = os.path.join("data", "raw", fname)
                if os.path.exists(save_path):
                    continue
                try:
                    download_filing(url, save_path)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
        except Exception as e:
            print(f"Error fetching filings for {company}: {e}")
