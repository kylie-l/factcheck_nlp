"""
Label claim–evidence pairs using a local Ollama model (LLaMA, Mistral, etc.) with resume support.

Input JSONL format:
{
  "claim": "...",
  "evidence": [ {"sentence": "..."} , ... ]
}
"""

import json
import os
import subprocess
import time
import argparse
from tqdm import tqdm
import re

# ========== CONFIG ==========
DEFAULT_MODEL = "llama3.1"
SYSTEM_PROMPT = (
    "You are an expert fact-checking assistant.\n"
    "Given a claim and its evidence, assign a support score from 1 to 5.\n\n"
    "1 = Strongly refutes the claim.\n"
    "2 = Weakly refutes the claim.\n"
    "3 = Not enough information to decide.\n"
    "4 = Weakly supports the claim.\n"
    "5 = Strongly supports the claim.\n\n"
    "Return ONLY a JSON object: {\"score\": <number>, \"rationale\": \"...\"}.\n"
    "The rationale must be concise (1–2 sentences)."
)

# ========== Utilities ==========
def make_prompt(claim, evidence_sentences):
    evidence_text = " ".join(e.get("sentence", "") for e in evidence_sentences).strip()
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Answer in JSON."
    )

WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

def parse_json_output(text):
    """
    Robust extractor for a 1–5 support score with multiple fallbacks.
    """

    if not text or not text.strip():
        return {"score": 3, "rationale": "Empty output."}

    # 1. Try exact JSON
    try:
        j = json.loads(text)
        s = int(j.get("score", 3))
        if 1 <= s <= 5:
            return j
    except:
        pass

    # 2. Extract JSON substring
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != -1:
            j = json.loads(text[start:end])
            s = int(j.get("score", 3))
            if 1 <= s <= 5:
                return j
    except:
        pass

    # 3. JSON-like key
    m = re.search(r'"score"\s*:\s*(\d)', text)
    if m:
        s = int(m.group(1))
        if 1 <= s <= 5:
            return {"score": s, "rationale": "Extracted from JSON-like pattern."}

    # 4. Natural language patterns
    patterns = [
        r"score is (\d)",
        r"score[:=]\s*(\d)",
        r"rated\s*(\d)",
        r"rating[:=]\s*(\d)",
        r"\bI would give this a (\d)\b",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            s = int(m.group(1))
            if 1 <= s <= 5:
                return {"score": s, "rationale": f"Extracted via pattern '{p}'."}

    # 5. Any standalone digit
    any_digit = re.findall(r"\b([1-5])\b", text)
    if any_digit:
        s = int(any_digit[0])
        return {"score": s, "rationale": "Extracted first valid standalone digit."}

    # 6. Word numbers
    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            return {"score": num, "rationale": f"Matched word-number '{word}'."}

    # 7. Final fallback
    return {"score": 3, "rationale": "Failed all parsing; defaulting to 3."}

# ========== Ollama interface ==========
def query_ollama(model, prompt, retries=3):
    for _ in range(retries):
        try:
            res = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=90,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except subprocess.TimeoutExpired:
            print("⏱️ Timeout, retrying...")
        time.sleep(2)
    return ""

# ========== Resume Logic ==========
def load_existing(outfile):
    done = set()
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done.add(item.get("claim", ""))
                exce
