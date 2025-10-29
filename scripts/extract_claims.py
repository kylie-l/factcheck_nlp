import os, re, json, argparse
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
from config.env_keywords import ENV_KEYWORDS

# Download NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")

# --- Measurable or commitment patterns ---
# CLAIM_PATTERNS = [
#     r"\d+%",                           # percentages
#     r"\bby\s+20\d{2}\b",               # "by 2030"
#     r"\bin\s+20\d{2}\b",               # "in 2024"
#     r"\b(before|through|until)\s+20\d{2}\b",  # time-related phrases
#     r"\b(increase|reduce|decrease|achieve|target|commit|reach|cut|eliminate|offset|goal|aim|improve)\b"
# ]

CLAIM_PATTERNS = [
    # --- Numeric patterns tied to environmental context ---
    r"\b(?:reduced|cut|decreased|lowered|offset|saved|captured|avoided)\b[^.]{0,50}\b\d+%?\b",   # e.g., reduced by 20%
    r"\b\d+%?\b[^.]{0,50}\b(?:reduction|decrease|cut|offset|savings|lower|lowering)\b",         # e.g., 25% reduction
    r"\b(?:emissions|energy|waste|water|renewable|recycling|carbon|footprint|solar|fuel)\b[^.]{0,50}\b\d+%?\b",  # metric first or last
    r"\bnet[-\s]?zero\b",                     # "net zero" or "net-zero"
    r"\bcarbon[-\s]?neutral\b",               # "carbon neutral" or "carbon-neutral"
    r"\bzero[-\s]?waste\b",                   # "zero waste"
    r"\b100% renewable\b",                    # "100% renewable"
    r"\bby\s+20\d{2}\b",                      # time-bound commitments ("by 2030")
    r"\b(in|before|after|until)\s+20\d{2}\b", # temporal phrasing
]



def extract_text_from_html(path):
    """Convert HTML to clean text."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def extract_text_from_txt(path):
    """Load plain text (for CSR reports)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def contains_env_keyword(sentence):
    """Check if the sentence contains a standalone environmental keyword."""
    s = re.sub(r"[^a-z\s]", " ", sentence.lower())
    for k in ENV_KEYWORDS:
        if re.search(rf"\b{re.escape(k)}\b", s):
            return True
    return False


def contains_claim_pattern(sentence):
    """Check if the sentence contains measurable or numeric language."""
    return any(re.search(pat, sentence, re.I) for pat in CLAIM_PATTERNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract environmental claims from text or HTML files.")
    parser.add_argument("--input", required=True, help="Input directory (e.g., data/raw or data/reports_text)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUT_PATH = args.output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".html", ".txt"))]
    output = []

    for fname in tqdm(files, desc="Processing files"):
        path = os.path.join(INPUT_DIR, fname)
        # Extract company and year cleanly from filenames like Amazon_2024_report.txt
        match = re.match(r"([A-Za-z]+)_(20\d{2})_", fname)
        if match:
            company, filing = match.groups()
        else:
            # fallback if pattern doesn't match
            parts = re.split(r"[_-]", fname.replace(".html", "").replace(".txt", ""))
            company = parts[0] if parts else "unknown"
            filing = parts[1] if len(parts) > 1 else "unknown"

        try:
            text = extract_text_from_html(path) if fname.endswith(".html") else extract_text_from_txt(path)

            for para in text.split("\n"):
                sentences = sent_tokenize(para)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent.split()) < 5 or len(sent.split()) > 60:
                        continue
                    if contains_env_keyword(sent) and contains_claim_pattern(sent):
                        output.append({
                            "company": company,
                            "filing": filing,
                            "claim_text": sent,
                            "source_file": fname
                        })

        except Exception as e:
            print(f"❌ Error parsing {fname}: {e}")

    # --- NEW: Drop duplicates by claim_text ---
    unique_output = []
    seen = set()
    for record in output:
        key = (record["company"], record["claim_text"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique_output.append(record)


    num_dropped = len(output) - len(unique_output)

    # --- Save cleaned JSONL ---
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for record in unique_output:
            f.write(json.dumps(record) + "\n")

    print(f"\n✅ Extracted {len(unique_output)} unique environmental claims from {len(files)} files.")
    print(f"🗑️ Dropped {num_dropped} duplicate claims.")
    print(f"Saved to {OUT_PATH}")
