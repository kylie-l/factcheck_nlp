#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List
import os
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

# NOW import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import TrainerCallback


# ============================================================
# Loss tracker callback
# ============================================================

class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)


# ============================================================
# Dataset
# ============================================================

@dataclass
class ClaimDataset(Dataset):
    items: List[Dict]
    tokenizer: AutoTokenizer
    max_length: int

    def __getitem__(self, idx):
        item = self.items[idx]
        claim = item["claim"]
        evidence = " ".join(ev["sentence"] for ev in item.get("evidence", []))
        text = f"Claim: {claim}\nEvidence: {evidence}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        label = float(item["label"])
        enc["labels"] = label
        return {k: torch.tensor(v) for k, v in enc.items()}

    def __len__(self):
        return len(self.items)


# ============================================================
# Metrics
# ============================================================

def ordinal_metrics(preds, gold):
    abs_err = np.abs(preds - gold)
    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    within1 = (abs_err <= 1).mean()
    return abs_err, mae, rmse, within1


def print_eval_report(preds, gold):
    preds_round = np.clip(np.round(preds), 1, 5).astype(int)

    print("\n=== Rounded Classification Metrics ===")
    print(f"Accuracy:     {accuracy_score(gold, preds_round):.4f}")
    print(f"Macro F1:     {f1_score(gold, preds_round, average='macro'):.4f}")
    print("\nReport:\n", classification_report(gold, preds_round))

    abs_err, mae, rmse, within1 = ordinal_metrics(preds_round, gold)

    print("\n=== Ordinal Metrics ===")
    print(f"MAE:          {mae:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"Within-1 acc: {within1:.4f}")

    print("\nError distance distribution:")
    for d in range(5):
        count = np.sum(abs_err == d)
        print(f" diff {d}: {count} ({count / len(abs_err):.3f})")

    print("\nPer-class mean absolute error (by gold label):")
    print("Label | Mean | Support")
    print("------|------|--------")
    for label in range(1, 6):
        idx = np.where(gold == label)[0]
        if len(idx) == 0:
            continue
        mean_err = abs_err[idx].mean()
        print(f"    {label} | {mean_err:.3f} | {len(idx)}")


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)

            if "llm_score" in obj:
                obj["label"] = obj["llm_score"]
            elif "gold_label" in obj:
                obj["label"] = obj["gold_label"]

            if obj["label"] in [1, 2, 3, 4, 5]:
                items.append(obj)

    return items


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="models/modernbert-regressor")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print("=== ModernBERT Regression Training ===")
    print(f"Train file : {args.train_file}")
    print(f"Gold file  : {args.gold_file}")
    print(f"Model      : {args.model_name}")
    print(f"Output dir : {args.output_dir}")
    print("============================================================")

    train_items = load_jsonl(args.train_file)
    gold_items  = load_jsonl(args.gold_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load regression head
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=True,
    )

    train_dataset = ClaimDataset(train_items, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )

    loss_tracker = LossTrackerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[loss_tracker],
    )

    trainer.train()
    print("✓ Training complete. Model saved to", args.output_dir)


    # ============================================================
    # Plot loss curve
    # ============================================================
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(loss_tracker.steps, loss_tracker.losses, label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (ModernBERT)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve to {out_path}")


    # ============================================================
    # Evaluation — FIXED DEVICE MISMATCH
    # ============================================================
    print("\n=== Evaluating on gold set ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    gold = []

    for item in tqdm(gold_items):
        claim = item["claim"]
        evidence = " ".join(ev["sentence"] for ev in item.get("evidence", []))
        text = f"Claim: {claim}\nEvidence: {evidence}"

        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # 🔥 CRITICAL FIX — move inputs to same device as model
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(**enc)

        score = output.logits.squeeze().item()
        preds.append(score)
        gold.append(int(item["label"]))

    preds = np.array(preds)
    gold = np.array(gold)

    print_eval_report(preds, gold)


if __name__ == "__main__":
    main()