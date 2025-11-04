"""
Label claim–evidence pairs using a local Ollama model (LLaMA, Mistral, etc.) with resume support.

Each input JSONL must have:
{
  "claim": "...",
  "evidence": [ {"sentence": "..."} , ... ]
}

Usage:
  python ollama_labeling_resume.py \
      --in data/processed/paired_dataset_cleaned.jsonl \
      --out data/processed/paired_dataset_labeled_ollama.jsonl \
      --model llama3.1
"""

import json
import os
import subprocess
import time
import argparse
from tqdm import tqdm

# ========== CONFIG ==========
DEFAULT_MODEL = "llama3.1"
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. "
    "Given a claim and its evidence, decide if the evidence SUPPORTS, REFUTES, "
    "or provides NOT ENOUGH INFO to verify the claim. "
    "Return JSON with keys: 'label' and 'rationale'. "
    "The rationale should be concise (1–2 sentences)."
)

# ---------- Utilities ----------
def make_prompt(claim, evidence_sentences):
    evidence_text = " ".join(e.get("sentence", "") for e in evidence_sentences).strip()
    return f"{SYSTEM_PROMPT}\n\nClaim: {claim}\n\nEvidence:\n{evidence_text}\n\nAnswer in JSON."

def parse_json_output(text):
    """Extract valid JSON if model adds extra text."""
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return {"label": "NOT ENOUGH INFO", "rationale": "Model output unparseable or missing."}

# ---------- Ollama interface ----------
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

# ---------- Resume helpers ----------
def load_existing(outfile):
    """Return dict of already labeled items by claim text."""
    done = {}
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done[item.get("claim", "")] = True
                except json.JSONDecodeError:
                    continue
    return done

# ---------- Main ----------
def main(args):
    with open(args.infile, "r") as f:
        data = [json.loads(line) for line in f]

    done = load_existing(args.outfile)
    print(f"🔁 Resuming from {len(done)} labeled pairs")

    with open(args.outfile, "a") as out_f:
        for item in tqdm(data, desc="Labeling with Ollama", unit="pair"):
            claim = item.get("claim", "")
            if claim in done:
                continue  # skip already done

            evidence = item.get("evidence", [])
            prompt = make_prompt(claim, evidence)

            raw_output = query_ollama(args.model, prompt)
            result = parse_json_output(raw_output)

            item["label"] = result["label"]
            item["rationale"] = result["rationale"]

            out_f.write(json.dumps(item) + "\n")
            out_f.flush()

    print("✅ Labeling complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="Input JSONL file")
    parser.add_argument("--out", dest="outfile", required=True, help="Output JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name (default: llama3.1)")
    args = parser.parse_args()
    main(args)
