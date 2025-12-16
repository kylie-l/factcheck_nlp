#!/usr/bin/env python3
"""
ModernBERT verifier: train on silver labels (1–5) and evaluate on gold set.

- Train file: JSONL with fields like:
    {
      "id": "claim_00001",
      "claim": "...",
      "company": "Amazon",
      "evidence": [
        {"sentence": "...", "similarity": 0.6},
        ...
      ],
      "llm_score": 3   # silver label (1–5)
    }

- Gold file: JSONL with same shape but with
    "gold_label": 1–5    # human label

We:
  * Build text = "Company: ... Claim: ... Evidence: ..."
  * Train ModernBERT (5-way classification) on silver labels
  * Evaluate on gold labels
"""

import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)


# --------------------------------------------------------------------
# Helpers for text construction and label extraction
# --------------------------------------------------------------------

VALID_LABELS = {1, 2, 3, 4, 5}


def build_evidence_text(evidence: Any) -> str:
    """
    Convert the 'evidence' field into a single text string.
    Supports:
      - list of dicts with 'sentence' or 'text'
      - plain strings
    """
    if isinstance(evidence, str):
        return evidence.strip()

    if isinstance(evidence, list):
        sentences = []
        for e in evidence:
            if isinstance(e, dict):
                sent = e.get("sentence") or e.get("text") or ""
            else:
                sent = str(e)
            sent = sent.strip()
            if sent:
                sentences.append(sent)
        return " || ".join(sentences)

    return str(evidence)


def make_input_text(item: Dict[str, Any]) -> str:
    """
    Build the classifier input text from claim + company + evidence.
    """
    company = (item.get("company") or "").strip()
    claim = (item.get("claim") or item.get("claim_text") or "").strip()
    evidence_text = build_evidence_text(item.get("evidence", ""))

    parts = []
    if company:
        parts.append(f"Company: {company}")
    parts.append(f"Claim: {claim}")
    parts.append(f"Evidence: {evidence_text}")

    return "\n".join(parts).strip()


def extract_label(item: Dict[str, Any], keys: List[str]) -> Optional[int]:
    """
    Try multiple label keys in order and return an integer label 1–5 or None.
    """
    for key in keys:
        if key in item and item[key] is not None:
            try:
                val = int(item[key])
                if val in VALID_LABELS:
                    return val
            except Exception:
                continue
    return None


# --------------------------------------------------------------------
# Torch Dataset
# --------------------------------------------------------------------

class ClaimEvidenceDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], label_key_order: List[str]):
        self.texts = []
        self.labels = []

        for obj in items:
            lbl = extract_label(obj, label_key_order)
            if lbl is None:
                continue
            text = make_input_text(obj)
            if not text:
                continue

            # Store labels as 0–4 internally (ModernBERT classifier indices)
            self.texts.append(text)
            self.labels.append(lbl - 1)

        self.labels = np.array(self.labels, dtype=int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": int(self.labels[idx]),
        }


# --------------------------------------------------------------------
# Metrics (classification + ordinal)
# --------------------------------------------------------------------

def print_full_metrics(pred_labels_15: np.ndarray, gold_labels_15: np.ndarray):
    """
    pred_labels_15, gold_labels_15: arrays of integers in [1..5]
    """
    labels = sorted(set(gold_labels_15.tolist()))
    preds = pred_labels_15
    gold = gold_labels_15

    # Classification metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        gold, preds, labels=labels, zero_division=0
    )

    print("\n=== Per-Class Metrics ===")
    print("Label | Precision | Recall | F1 | Support")
    print("------|-----------|--------|----|---------")
    for l, p, r, f, s in zip(labels, prec, rec, f1, support):
        print(f"{l:5d} | {p:9.3f} | {r:6.3f} | {f:4.3f} | {s}")

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(gold, preds, average="micro", zero_division=0)
    weighted_f1 = f1_score(gold, preds, average="weighted", zero_division=0)

    print("\n=== Summary (Classification) ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}   <- best for imbalanced labels")
    print(f"Micro F1:     {micro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    # Ordinal metrics
    abs_err = np.abs(preds - gold)
    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    within1 = np.mean(abs_err <= 1)

    print("\n=== Ordinal Error Metrics (1–5 scale) ===")
    print(f"MAE:          {mae:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"Within-1 acc: {within1:.4f}   <- |pred - gold| \u2264 1")

    # Error distance distribution
    print("\nError distance distribution (absolute difference):")
    print("diff | count | proportion")
    print("-----|-------|-----------")
    for d in range(0, 5):
        count = int(np.sum(abs_err == d))
        prop = count / len(abs_err)
        print(f"{d:4d} | {count:5d} | {prop:9.3f}")

    # Per-class MAE
    print("\nPer-class mean absolute error (by gold label):")
    print("Label | Mean | Support")
    print("------|------|--------")
    for l in labels:
        idx = np.where(gold == l)[0]
        if len(idx) == 0:
            continue
        mean_mae = abs_err[idx].mean()
        print(f"{l:5d} | {mean_mae:4.3f} | {len(idx)}")


# --------------------------------------------------------------------
# Main training + evaluation
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT verifier (1–5) and evaluate on gold labels."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to silver-labeled JSONL (e.g., silver_labeled_granite.jsonl)",
    )
    parser.add_argument(
        "--gold-file",
        type=str,
        required=True,
        help="Path to gold-labeled JSONL with 64 examples (e.g., gold_labeled.jsonl)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="ModernBERT model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/modernbert-verifier",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=== ModernBERT Verifier Training ===")
    print(f"Train file : {args.train_file}")
    print(f"Gold file  : {args.gold_file}")
    print(f"Model      : {args.model_name}")
    print(f"Output dir : {args.output_dir}")
    print(f"Epochs     : {args.num_epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max length : {args.max_length}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load JSONL data
    # --------------------------------------------------------
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    train_items = load_jsonl(args.train_file)
    gold_items = load_jsonl(args.gold_file)

    print(f"Loaded {len(train_items)} training items.")
    print(f"Loaded {len(gold_items)} gold items.")

    # Silver labels: prefer llm_score, then gold_label, then label
    silver_label_keys = ["llm_score", "gold_label", "label"]
    gold_label_keys = ["gold_label"]

    train_dataset = ClaimEvidenceDataset(train_items, silver_label_keys)
    eval_dataset = ClaimEvidenceDataset(gold_items, gold_label_keys)

    print(f"Train dataset size (valid labels): {len(train_dataset)}")
    print(f"Eval  dataset size (valid labels): {len(eval_dataset)}")

    # --------------------------------------------------------
    # Tokenizer & model
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    # Lazy tokenization via Trainer: we just store "text" + "labels" in dataset;
    # Trainer will call tokenizer in a data_collator if we wrap it properly.
    # Easiest: map manually to tokenized dicts.

    def to_tokenized_dataset(ds: ClaimEvidenceDataset):
        texts = ds.texts
        labels = ds.labels
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        encodings["labels"] = labels
        # Wrap as simple torch Dataset of dicts
        class EncodedDataset(Dataset):
            def __len__(self):
                return len(labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(encodings["input_ids"][idx]),
                    "attention_mask": torch.tensor(encodings["attention_mask"][idx]),
                    "labels": torch.tensor(int(encodings["labels"][idx])),
                }

        return EncodedDataset()

    train_ds_enc = to_tokenized_dataset(train_dataset)
    eval_ds_enc = to_tokenized_dataset(eval_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --------------------------------------------------------
    # TrainingArguments + Trainer
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",   # we'll evaluate separately on the 64 gold
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_enc,
        eval_dataset=eval_ds_enc,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✓ Training complete. Model saved to {args.output_dir}")

    # --------------------------------------------------------
    # Evaluation on gold (64 examples)
    # --------------------------------------------------------
    print("\n=== Evaluating on gold set ===")
    preds_output = trainer.predict(eval_ds_enc)
    logits = preds_output.predictions
    pred_ids = np.argmax(logits, axis=-1)      # 0–4
    pred_labels_15 = pred_ids + 1             # back to 1–5
    gold_labels_15 = eval_dataset.labels + 1  # eval_dataset.labels is 0–4

    print_full_metrics(pred_labels_15, gold_labels_15)


if __name__ == "__main__":
    main()
