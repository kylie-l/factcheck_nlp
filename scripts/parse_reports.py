import os
import re
import fitz  # PyMuPDF
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

RAW_DIR = "data/reports"
TEXT_DIR = "data/reports_text"
COMPANIES_FILE = "companies.txt"

os.makedirs(TEXT_DIR, exist_ok=True)

# Download NLTK data (only first run)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# --- Helper functions ---
def load_companies(path):
    """Load normalized company names."""
    with open(path, "r") as f:
        return [re.sub(r'[^a-zA-Z0-9]', '', line.strip()).capitalize() for line in f if line.strip()]


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text.strip()


def extract_text_from_txt(txt_path):
    """Extract text from a text file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {txt_path}: {e}")
        return ""


def parse_filename(fname, companies):
    """Extract company and year from standardized filename."""
    parts = re.split(r"[_\-]", fname)
    if len(parts) < 2:
        return None, None

    # Try to find year (20xx)
    year = next((p for p in parts if re.fullmatch(r"20\d{2}", p)), None)

    # Company = first token that matches one in companies (case-insensitive)
    company = next((c for c in companies if c.lower() in fname.lower()), None)

    if not company or not year:
        return None, None
    return company, year


# --- Main processing ---
def main():
    companies = load_companies(COMPANIES_FILE)
    reports = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".pdf", ".txt"))]
    summary = []

    for fname in tqdm(reports, desc="Parsing reports"):
        company, year = parse_filename(fname, companies)
        if not company or not year:
            print(f"⚠️ Skipping {fname} — invalid prefix or unknown company/year")
            continue

        path = os.path.join(RAW_DIR, fname)
        out_name = f"{company}_{year}_report.txt"
        out_path = os.path.join(TEXT_DIR, out_name)

        # Extract raw text
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            text = extract_text_from_txt(path)

        if not text.strip():
            print(f"⚠️ No text extracted from {fname}")
            continue

        # --- Clean text before tokenization ---
        # Remove hyphenation splits and merge lines that aren't true breaks
        cleaned = re.sub(r'-\n', '', text)              # join hyphenated words split across lines
        cleaned = re.sub(r'\n+', '\n', cleaned)         # collapse multiple newlines
        cleaned = re.sub(r'(?<![.?!])\n(?![A-Z])', ' ', cleaned)  # join lines mid-sentence
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # normalize spaces

        # --- Sentence tokenize ---
        sentences = sent_tokenize(cleaned)
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]


        # Save each sentence on a new line
        with open(out_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        summary.append((fname, company, year, len(sentences)))

    # Print summary table
    print("\n✅ Finished parsing. Sentence-level text files saved to:", TEXT_DIR)
    if summary:
        print("\nSummary:")
        for fname, comp, year, count in summary:
            print(f"  {fname:<60}  →  {comp} ({year}) — {count} sentences")


if __name__ == "__main__":
    main()