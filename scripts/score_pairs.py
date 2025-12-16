#!/usr/bin/env python3
"""
Fact Verification Scoring Script (vLLM)

This script uses an LLM (via vLLM) as a judge to score claim–evidence pairs.

Task:
- Input: JSONL with fields like:
    {
      "id": "claim_00001",
      "claim": "...",
      "company": "Amazon",
      "evidence": [
        {"sentence": "..."},
        {"sentence": "..."},
        ...
      ]
    }

- Output: JSONL with added fields:
    {
      "id": "claim_00001",
      "claim": "...",
      "company": "Amazon",
      "evidence": [...],
      "llm_score": 2,
      "llm_rationale": "Short explanation...",
      "raw_response": "...full model text..."
    }

Scoring scale (1–5):
  1 = Strongly supports the claim
  2 = Weakly supports the claim
  3 = Not enough information to decide
  4 = Weakly refutes the claim
  5 = Strongly refutes the claim

If a gold CSV file is provided,
the script will evaluate the model's scores against the gold labels.

Gold CSV format (flexible):
- Must contain a column "id"
- Must contain either "score" or "label" with numeric values 1-5
"""

import argparse
import csv
import json
import re
from typing import Dict, List, Optional, Any
import subprocess

import numpy as np
import torch
from tqdm import tqdm
LLM = None
SamplingParams = None
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# PROMPT ENGINEERING
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert fact-checking assistant specializing in climate and "
    "sustainability claims from corporate reports and news. Your job is to read "
    "a claim and the provided evidence, and decide how strongly the evidence "
    "supports or refutes the claim."
)

FACT_VERIFICATION_RUBRIC = """
Scoring rubric (1-5):

1 = Strongly supports the claim.
    - Evidence directly and clearly states the claim is true.
    - No major caveats; details (numbers, dates, targets) match.

2 = Weakly supports the claim.
    - Evidence is generally consistent with the claim but is indirect,
      partially relevant, or missing some details.

3 = Not enough information.
    - Evidence is related to the topic or company but does NOT clearly
      confirm or refute the specific claim.
    - If the evidence is mostly background context or off-topic, choose 3.

4 = Weakly refutes the claim.
    - Evidence suggests the claim is unlikely or misleading but is indirect
      or missing some details.

5 = Strongly refutes the claim.
    - Evidence directly contradicts the claim or shows it is false
      (e.g., opposite numbers, missed targets, or explicit disagreement).

Important:
- If the evidence does not clearly support or refute the specific claim,
  choose 3 (Not enough information).
- Do NOT guess; stay faithful to the provided evidence only.
"""


def build_evidence_text(evidence: Any) -> str:
    """
    Convert the 'evidence' field (list of dicts or string) into a readable block of text.
    """
    if isinstance(evidence, str):
        return evidence.strip()

    if isinstance(evidence, list):
        sentences = []
        for e in evidence:
            sent = e.get("sentence") or e.get("text") or ""
            sent = sent.strip()
            if sent:
                sentences.append(sent)
        return " || ".join(sentences)

    # Fallback: just cast to string
    return str(evidence)


def create_prompt(claim: str, evidence_text: str, prompt_id: int = 0) -> str:
    """
    Create a fact-verification prompt for the LLM.

    Args:
        claim: The claim text
        evidence_text: Evidence sentences concatenated
        prompt_id: Variant selector (you can experiment with different prompts)

    Returns:
        Prompt string.
    """
    # Single main prompt variant for now; you can add more if desired.
    if prompt_id == 0:
        prompt = f"""
You are given a CLAIM and some EVIDENCE about a company's climate or sustainability actions.

Your task:
1. Decide how strongly the EVIDENCE supports or refutes the CLAIM.
2. Assign a single support score from 1 to 5 using the rubric below.
3. Provide a brief, 1–2 sentence rationale that explains your decision.
4. Output ONLY a valid JSON object.

{FACT_VERIFICATION_RUBRIC}

--------------------
CLAIM:
{claim}

--------------------
EVIDENCE:
{evidence_text}

--------------------
Output format (JSON only):
{{
  "score": <integer from 1 to 5>,
  "rationale": "<one or two concise sentences>"
}}

Remember:
- If the evidence is related but does not clearly prove or disprove the claim,
  choose score 3.
- Do not add any text outside the JSON object.
"""
    else:
        # You can add other variants here if you want to experiment.
        prompt = f"""
Use the same rubric as before, but be extra conservative.
If the evidence does not clearly support or refute, prefer score 3.

{FACT_VERIFICATION_RUBRIC}

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

Output JSON:
{{"score": <1-5>, "rationale": "<brief explanation>"}}
"""

    return prompt.strip()


# ---------------------------------------------------------------------------
# RESPONSE PARSING
# ---------------------------------------------------------------------------

WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_score_and_rationale(response_text: str) -> Dict[str, Any]:
    """
    Robustly extract a 1–5 score and rationale from the model's response.

    The model is *asked* to output JSON, but we defensively:
      1. Try direct JSON parse.
      2. Try extracting JSON substring.
      3. Try JSON-like `"score": 3` pattern.
      4. Regex patterns like "score is 4", "score: 2".
      5. First standalone digit 1–5.
      6. Word-number fallback ("one", "two", ...).
    If everything fails, default to score=3 and generic rationale.

    Returns:
        {"score": int, "rationale": str, "raw": str}
    """
    text = response_text or ""
    text = text.strip()
    if not text:
        return {
            "score": 3,
            "rationale": "Empty model output; defaulting to Not Enough Information.",
            "raw": response_text,
        }

    # 1. Try parsing as pure JSON
    try:
        obj = json.loads(text)
        s = obj.get("score")
        if s is not None:
            s_int = int(s)
            if 1 <= s_int <= 5:
                rationale = obj.get("rationale", "").strip()
                if not rationale:
                    rationale = "Model did not provide a rationale."
                return {"score": s_int, "rationale": rationale, "raw": response_text}
    except Exception:
        pass

    # 2. Try extracting JSON substring
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            snippet = text[start:end]
            obj = json.loads(snippet)
            s = obj.get("score")
            if s is not None:
                s_int = int(s)
                if 1 <= s_int <= 5:
                    rationale = obj.get("rationale", "").strip()
                    if not rationale:
                        rationale = "Model did not provide a rationale."
                    return {"score": s_int, "rationale": rationale, "raw": response_text}
    except Exception:
        pass

    # 3. JSON-like pattern: "score": 3
    m_json_like = re.search(r'"score"\s*:\s*(\d)', text)
    if m_json_like:
        s_int = int(m_json_like.group(1))
        if 1 <= s_int <= 5:
            return {
                "score": s_int,
                "rationale": "Extracted score from JSON-like pattern in model output.",
                "raw": response_text,
            }

    # 4. Patterns like "score is 4", "score: 2"
    patterns = [
        r"score is (\d)",
        r"score[:=]\s*(\d)",
        r"support score[:=]\s*(\d)",
        r"\bI would give this (?:claim|pair)?\s*a (\d)\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            s_int = int(m.group(1))
            if 1 <= s_int <= 5:
                return {
                    "score": s_int,
                    "rationale": f"Score extracted via pattern '{p}'.",
                    "raw": response_text,
                }

    # 5. Any standalone digit 1–5
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        s_int = int(digits[0])
        return {
            "score": s_int,
            "rationale": "Score extracted as first standalone digit 1–5.",
            "raw": response_text,
        }

    # 6. Word-number fallback ("one", "two", etc.)
    lower = text.lower()
    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            return {
                "score": num,
                "rationale": f"Score inferred from word-number '{word}'.",
                "raw": response_text,
            }

    # 7. Final fallback
    return {
        "score": 3,
        "rationale": "Failed to parse model output; defaulting to Not Enough Information (3).",
        "raw": response_text,
    }


# ---------------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------------

def read_claims_jsonl(filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read claim–evidence pairs from a JSONL file.
    """
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def write_results_jsonl(results: List[Dict[str, Any]], output_file: str):
    """
    Write results to a JSONL file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_gold_labels(gold_file: str) -> Dict[str, int]:
    gold = {}

    # Detect file type
    is_jsonl = gold_file.endswith(".jsonl") or gold_file.endswith(".jl")

    # JSONL loader
    if is_jsonl:
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("id")
                label = (
                    obj.get("gold_label") or
                    obj.get("label") or
                    obj.get("score")
                )
                if cid and label:
                    try:
                        s = int(label)
                        if 1 <= s <= 5:
                            gold[cid] = s
                    except ValueError:
                        continue
        return gold

    # CSV loader (fallback)
    with open(gold_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("id")
            label = row.get("gold_label") or row.get("label") or row.get("score")
            if cid and label:
                try:
                    s = int(label)
                    if 1 <= s <= 5:
                        gold[cid] = s
                except ValueError:
                    continue

    return gold
    
def run_ollama_single(model: str, prompt: str, retries: int = 3) -> str:
    """
    Query Ollama locally. Returns raw model output string.
    """
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"[Ollama] Non-zero exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            print("[Ollama] Timeout, retrying...")
        time.sleep(1)

    return ""



# ---------------------------------------------------------------------------
# BATCH INFERENCE
# ---------------------------------------------------------------------------

def process_chunk(
    chunk_data: List[Dict[str, Any]],
    model: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    use_chat_template: bool = True,
    system_message: Optional[str] = None,
    prompt_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Process a chunk of claim–evidence pairs through vLLM in batch.
    """
    input_list = []
    metadata_list = []

    for item in chunk_data:
        cid = str(item.get("id") or item.get("claim_id") or "")
        claim = (item.get("claim") or item.get("claim_text") or "").strip()
        evidence_raw = item.get("evidence", "")

        if not claim:
            metadata_list.append({
                **item,
                "llm_score": None,
                "llm_rationale": None,
                "raw_response": None,
                "error": "Missing claim text.",
            })
            input_list.append(None)
            continue

        evidence_text = build_evidence_text(evidence_raw)

        try:
            prompt = create_prompt(claim, evidence_text, prompt_id=prompt_id)
        except Exception as e:
            metadata_list.append({
                **item,
                "llm_score": None,
                "llm_rationale": None,
                "raw_response": None,
                "error": f"Prompt creation error: {e}",
            })
            input_list.append(None)
            continue

        if not prompt.strip():
            metadata_list.append({
                **item,
                "llm_score": None,
                "llm_rationale": None,
                "raw_response": None,
                "error": "Empty prompt.",
            })
            input_list.append(None)
            continue

        if use_chat_template:
            sys_msg = system_message or SYSTEM_MESSAGE
            chat_input = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ]
            input_list.append(chat_input)
        else:
            input_list.append(prompt)

        metadata_list.append({
            **item,
            "error": None,
        })

    # Filter valid
    valid_indices = []
    valid_inputs = []
    for i, inp in enumerate(input_list):
        if inp is not None:
            valid_indices.append(i)
            valid_inputs.append(inp)

    if not valid_inputs:
        return metadata_list

    # Apply chat template
    if use_chat_template:
        formatted_inputs = [
            tokenizer.apply_chat_template(
                user_input,
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=True,
            )
            for user_input in valid_inputs
        ]
    else:
        formatted_inputs = valid_inputs

    # Generate with vLLM
    outputs = model.generate(formatted_inputs, sampling_params)
    output_texts = [o.outputs[0].text.strip() for o in outputs]

    # Merge results
    results = []
    out_idx = 0
    for i, meta in enumerate(metadata_list):
        if i in valid_indices:
            raw_resp = output_texts[out_idx]
            out_idx += 1
            parsed = parse_score_and_rationale(raw_resp)
            enriched = {
                **meta,
                "llm_score": parsed["score"],
                "llm_rationale": parsed["rationale"],
                "raw_response": parsed["raw"],
            }
            results.append(enriched)
        else:
            results.append(meta)

    return results

def process_ollama_items(
    data: List[Dict[str, Any]],
    model_name: str,
    prompt_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Sequential inference path for Ollama (no batching).
    """
    results = []

    for item in tqdm(data, desc="Scoring pairs (Ollama)", unit="pair"):
        claim = item.get("claim", "").strip()
        evidence_raw = item.get("evidence", "")
        evidence_text = build_evidence_text(evidence_raw)

        prompt = create_prompt(claim, evidence_text, prompt_id=prompt_id)

        raw = run_ollama_single(model_name, prompt)
        parsed = parse_score_and_rationale(raw)

        enriched = {
            **item,
            "llm_score": parsed["score"],
            "llm_rationale": parsed["rationale"],
            "raw_response": parsed["raw"],
            "error": None,
        }
        results.append(enriched)

    return results



# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_predictions(results: List[Dict[str, Any]], gold: Dict[str, int]) -> Dict[str, float]:
    """
    Evaluate model predictions against gold labels.

    Only examples whose 'id' appears in gold are used.

    Metrics:
      - MAE
      - RMSE
      - Exact match accuracy
      - Macro-averaged F1 (over 1–5)
    """
    preds = []
    truths = []

    for r in results:
        cid = str(r.get("id") or r.get("claim_id") or "")
        if not cid:
            continue
        if cid not in gold:
            continue
        score = r.get("llm_score")
        if score is None:
            continue
        try:
            s = int(score)
        except Exception:
            continue
        if 1 <= s <= 5:
            preds.append(s)
            truths.append(gold[cid])

    if not preds:
        return {
            "num_eval_examples": 0,
            "error": "No overlapping ids between predictions and gold labels.",
        }

    preds = np.array(preds, dtype=float)
    truths = np.array(truths, dtype=float)

    mae = float(np.mean(np.abs(preds - truths)))
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    exact = float(np.mean(preds == truths))

    # Macro F1 over 5 classes
    f1_per_class = []
    for c in [1, 2, 3, 4, 5]:
        tp = np.sum((preds == c) & (truths == c))
        fp = np.sum((preds == c) & (truths != c))
        fn = np.sum((preds != c) & (truths == c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1 = 0.0
        elif tp == 0 and fp == 0 and fn == 0:
            # Class not present in either predictions or truths; skip
            continue
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1_per_class.append(f1)

    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0

    return {
        "num_eval_examples": len(preds),
        "mae": mae,
        "rmse": rmse,
        "exact_match_accuracy": exact,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fact verification scoring with vLLM or Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="fact_scores.jsonl")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gold-file", type=str, default=None)

    # Model config
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)

    # Inference settings
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.add_argument("--system-message", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-id", type=int, default=0)

    # NEW backend selector
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "ollama"],
        help="Inference backend: vllm (GPU) or ollama (local CPU)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"Fact Verification Scoring (Backend: {args.backend})")
    print("=" * 70)
    print(f"Input file : {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model      : {args.model_name}")
    print(f"Chunk size : {args.chunk_size}")
    if args.limit:
        print(f"Limit      : {args.limit}")
    if args.evaluate:
        print(f"Gold file  : {args.gold_file}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load input data first (works for both backends)
    # -------------------------------------------------------------
    print("\n[1/4] Loading claim–evidence pairs...")
    data = read_claims_jsonl(args.input_file, limit=args.limit)
    print(f"Loaded {len(data)} items.")

    # -------------------------------------------------------------
    # BACKEND = OLLAMA  (skip tokenizer + vLLM initialization)
    # -------------------------------------------------------------
    if args.backend == "ollama":
        print("\n[2/4] Running inference with Ollama...")
        all_results = process_ollama_items(
            data,
            model_name=args.model_name,
            prompt_id=args.prompt_id,
        )
        # After inference, jump to saving & evaluation
        write_results_jsonl(all_results, args.output_file)
        print(f"\n✓ Saved {len(all_results)} scored pairs to {args.output_file}")

        # Optional evaluation
        if args.evaluate and args.gold_file:
            gold = load_gold_labels(args.gold_file)
            metrics = evaluate_predictions(all_results, gold)
            print("\n===== Evaluation =====")
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("======================")

        # Summary
        num_with_scores = sum(1 for r in all_results if r.get("llm_score") is not None)
        num_errors = sum(1 for r in all_results if r.get("error") is not None)
        print("\n===== Summary =====")
        print(f"Total items          : {len(all_results)}")
        print(f"With LLM scores      : {num_with_scores}")
        print(f"Items with error flag: {num_errors}")
        print("====================")
        return

    # -------------------------------------------------------------
    # BACKEND = VLLM  (load tokenizer + vLLM model)
    # -------------------------------------------------------------
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "ERROR: vLLM is not installed. "
            "You selected --backend vllm, but this machine does not support it.\n"
            "Install vLLM on a Linux GPU machine.\n"
        )

    print("\n[2/4] Loading tokenizer (vLLM)...")
    tok_kwargs = {"trust_remote_code": True}
    if args.cache_dir:
        tok_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    print("✓ Tokenizer loaded.")

    print("\n[3/4] Loading vLLM model...")
    available_gpus = torch.cuda.device_count()
    tp_size = args.tensor_parallel_size or max(1, available_gpus)
    if tp_size > available_gpus and available_gpus > 0:
        print(f"Warning: requested tensor_parallel_size={tp_size} but only {available_gpus} GPUs available.")
        tp_size = available_gpus

    model_kwargs = {"model": args.model_name, "tensor_parallel_size": tp_size}
    if args.cache_dir:
        model_kwargs["download_dir"] = args.cache_dir

    model = LLM(**model_kwargs)
    print("✓ Model loaded.")

    print("\n[4/4] Setting sampling parameters...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    print("✓ Sampling params configured.")

    # -------------------------------------------------------------
    # vLLM INFERENCE LOOP
    # -------------------------------------------------------------
    all_results = []
    print("\n[Inference] Using vLLM...")
    with tqdm(total=len(data), desc="Scoring pairs", unit="pair") as pbar:
        for i in range(0, len(data), args.chunk_size):
            chunk = data[i : i + args.chunk_size]
            chunk_results = process_chunk(
                chunk,
                model,
                tokenizer,
                sampling_params,
                use_chat_template=args.use_chat_template,
                system_message=args.system_message,
                prompt_id=args.prompt_id,
            )
            all_results.extend(chunk_results)
            pbar.update(len(chunk))

    # -------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------
    write_results_jsonl(all_results, args.output_file)
    print(f"\n✓ Saved {len(all_results)} scored pairs to {args.output_file}")

    # -------------------------------------------------------------
    # OPTIONAL EVALUATION
    # -------------------------------------------------------------
    if args.evaluate and args.gold_file:
        gold = load_gold_labels(args.gold_file)
        metrics = evaluate_predictions(all_results, gold)

        print("\n===== Evaluation =====")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("======================")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    num_with_scores = sum(1 for r in all_results if r.get("llm_score") is not None)
    num_errors = sum(1 for r in all_results if r.get("error") is not None)

    print("\n===== Summary =====")
    print(f"Total items          : {len(all_results)}")
    print(f"With LLM scores      : {num_with_scores}")
    print(f"Items with error flag: {num_errors}")
    print("====================")


if __name__ == "__main__":
    main()
