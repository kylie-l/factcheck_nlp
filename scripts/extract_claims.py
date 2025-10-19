import os, re, json, argparse
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk

# Download NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")

# --- Environmental ESG keywords only ---
ENV_KEYWORDS = [
    "climate change", "carbon", "greenhouse gas", "ghg", "emission",
    "renewable", "solar", "wind", "geothermal", "hydrogen",
    "recycling", "waste reduction", "water use", "water conservation",
    "biodiversity", "deforestation", "net zero",
    "energy efficiency", "sustainability report", "environmental impact",
    "pollution", "scope 1", "scope 2", "scope 3", "green", "sustainability",
    "zero waste", "carbon neutral", "low-carbon", "clean energy", "decarbonize",
    "eco-friendly"
]

# --- Measurable or commitment patterns ---
CLAIM_PATTERNS = [
    r"\d+%",                           # percentages
    r"\bby\s+20\d{2}\b",               # "by 2030"
    r"\bin\s+20\d{2}\b",               # "in 2024"
    r"\b(before|through|until)\s+20\d{2}\b",  # time-related phrases
    r"\b(increase|reduce|decrease|achieve|target|commit|reach|cut|eliminate|offset|goal|aim|improve)\b"
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
                # Split into sentences first (this will handle long paragraphs)
                sentences = sent_tokenize(para)
                for sent in sentences:
                    sent = sent.strip()
                    # Skip overly long or short sentences
                    if len(sent.split()) < 5 or len(sent.split()) > 60:
                        continue
                    # Only keep sentences that are actual environmental claims
                    if contains_env_keyword(sent) and contains_claim_pattern(sent):
                        output.append({
                            "company": company,
                            "filing": filing,
                            "claim_text": sent,
                            "source_file": fname
                        })

        except Exception as e:
            print(f"❌ Error parsing {fname}: {e}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for record in output:
            f.write(json.dumps(record) + "\n")

    print(f"\n✅ Extracted {len(output)} environmental claims from {len(files)} files.")
    print(f"Saved to {OUT_PATH}")
