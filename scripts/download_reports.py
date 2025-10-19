import os, re, requests, time
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("SERPAPI_KEY")
SEARCH_URL = "https://serpapi.com/search.json"

COMPANIES_FILE = "companies.txt"
OUT_DIR = "data/reports"
CURRENT_YEAR = datetime.now().year
os.makedirs(OUT_DIR, exist_ok=True)

# --- Helpers ---
def normalize_filename(name):
    """Make filenames safe and consistent."""
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return re.sub(r"_+", "_", name).strip("_")

def report_exists(company, year):
    """Check if a report already exists for a company/year."""
    prefix = f"{company}_{year}".lower()
    for f in os.listdir(OUT_DIR):
        if f.lower().startswith(prefix) and f.lower().endswith(".pdf"):
            return True
    return False

def find_report_pdf(company, year):
    """Query SerpAPI for the sustainability or CSR report."""
    if not API_KEY:
        print("❌ Missing SERPAPI_KEY environment variable. Set it with: export SERPAPI_KEY='your_key_here'")
        return []

    query = f"{company} {year} sustainability report filetype:pdf"
    params = {"engine": "google", "q": query, "api_key": API_KEY, "num": 5}

    try:
        r = requests.get(SEARCH_URL, params=params, timeout=20)
        if r.status_code == 401:
            print(f"⚠️ Unauthorized (401): Check your SerpAPI key or quota.")
            return []
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Request failed for {company} ({year}): {e}")
        return []

    results = r.json().get("organic_results", [])
    pdf_links = [res.get("link") for res in results if res.get("link", "").lower().endswith(".pdf")]
    return pdf_links[:2]  # top 2 results

def download_pdf(url, company, year):
    """Download a PDF if not already downloaded."""
    fname = os.path.basename(url)
    if not fname.lower().endswith(".pdf"):
        fname = f"{company}_{year}_report.pdf"

    fname = f"{company}_{year}_{normalize_filename(fname)}"
    path = os.path.join(OUT_DIR, fname)

    if os.path.exists(path):
        return path

    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return path
        else:
            print(f"⚠️ Skipping invalid content for {company} ({url})")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
    return None

# --- Main ---
if __name__ == "__main__":
    with open(COMPANIES_FILE) as f:
        companies = [c.strip() for c in f if c.strip()]

    downloaded, skipped = [], []

    for company in tqdm(companies, desc="Downloading CSR PDFs"):
        clean_name = normalize_filename(company)

        # Skip if already downloaded
        if report_exists(clean_name, CURRENT_YEAR):
            skipped.append(company)
            continue

        # Try current year first, then fallback to previous
        links = find_report_pdf(company, CURRENT_YEAR)
        if not links:
            links = find_report_pdf(company, CURRENT_YEAR - 1)

        if not links:
            print(f"⚠️ No report found for {company} ({CURRENT_YEAR} or {CURRENT_YEAR-1})")
            continue

        for link in links:
            pdf_path = download_pdf(link, clean_name, CURRENT_YEAR)
            if pdf_path:
                print(f"✅ Downloaded {pdf_path}")
                downloaded.append(company)
            time.sleep(2)  # avoid SerpAPI rate limit

    print("\n✅ Done.")
    print(f"Downloaded: {len(downloaded)} | Skipped existing: {len(skipped)}")
    if skipped:
        print("Skipped:", ", ".join(skipped))
