# Verifying Corporate Sustainability Claims with Ordinal NLP Models

This project investigates whether corporate environmental sustainability claims can be meaningfully verified against independent external reporting using natural language processing. I develop an end-to-end pipeline that extracts claims from corporate sustainability reports, retrieves relevant evidence from news sources, and predicts graded veracity scores using an ordinal regression model.

Rather than framing verification as a binary or categorical task, this project models veracity on a **1–5 ordinal scale**, capturing partial support, uncertainty, and weak refutation—an important distinction in domains where evidence is often fragmented or indirect.

This repository contains the full data processing, annotation, modeling, and evaluation pipeline described in the accompanying final report.

---

## 📌 Final Pipeline (Start Here)

The files listed below correspond directly to the **final pipeline used in the report**, ordered by execution flow.

### 1. Claim Collection from Sustainability Reports

- `scripts/download_reports.py`  
  Downloads publicly available corporate sustainability reports.

- `scripts/parse_reports.py`  
  Extracts raw text from reports and segments documents into sentence-level text.

- `scripts/extract_claims.py`  
  Identifies candidate environmental sustainability claims using keyword-based and pattern-based heuristics.

---

### 2. Evidence Collection and Standardization

- `scripts/standardize_evidence.py`  
  Collects and normalizes external news articles from the Guardian and GDELT, standardizing text and metadata.

- `scripts/combine_claims_evidence.py`  
  Matches claims to candidate evidence sentences using semantic similarity and company-based filtering.

---

### 3. Dataset Cleaning and Refinement

- `scripts/clean_pairs.py`  
  Removes malformed or low-quality claim–evidence pairs.

- `scripts/clean_2.py`  
  Applies additional filtering to remove headings, table artifacts, slide text, navigational content, and non-propositional claims to better align the dataset with claim-level factual verification.

---

### 4. Annotation and Dataset Splits

- `scripts/sample_for_annotation.py`  
  Randomly samples claim–evidence pairs for manual gold annotation.

- `scripts/split_train_eval.py`  
  Splits the dataset into LLM-labeled training data and a held-out gold evaluation set.

- `scripts/score_pairs.py`  
  Assigns ordinal veracity scores (1–5) using the Granite large language model and generates short rationales.

---

### 5. Modeling and Baselines

- `scripts/models/modernbert_regressor.py`  
  Trains and evaluates a ModernBERT-based ordinal regression model.

- `scripts/models/baseline_ordinal.py`  
  Implements random and majority ordinal baselines for comparison.

---

## 📂 Key Processed Data Files

The following files contain the finalized datasets used in the report:

- `data/processed/silver_labeled_granite.jsonl`  
  LLM-labeled claim–evidence pairs used for training.

- `data/processed/gold_labeled.jsonl`  
  Manually annotated gold evaluation set (64 instances), used exclusively for evaluation.

- `data/processed/annotated_sample_labeled.csv`  
  CSV version of sampled annotations used during development and validation.

---

## 🧪 Methods Overview

- **Evidence Retrieval:** Dense sentence-level retrieval using `all-MiniLM-L6-v2`
- **Annotation Strategy:** Hybrid manual + LLM-based ordinal labeling
- **Model:** ModernBERT regression model predicting continuous scores on a 1–5 scale
- **Evaluation:** Accuracy, Macro-F1 (rounded), MAE, RMSE, and within-one accuracy on a held-out gold set

---

## 📈 Key Findings

- Ordinal regression outperforms random and majority baselines across all ordinal metrics.
- Most model errors fall within one point of the gold label, indicating calibrated predictions.
- Performance is constrained more by evidence availability and label ambiguity than model capacity.

---

## 🗂 Repository Notes

This repository also contains scripts and files from earlier experiments (e.g., classification-based models, alternative LLMs, prompt variants). These are retained for transparency and reproducibility but are **not required** to reproduce the final results reported in the paper.

---

## 🔮 Future Directions

Potential extensions include:
- Expanding evidence sources to include regulatory filings or investigative reporting
- Incorporating source credibility into evidence aggregation
- Extending sentence-level verification to document-level or multi-source reasoning
- Increasing expert-driven gold annotations for stronger evaluation of extreme labels

---

## 📄 Report and Code

The full project report provides detailed motivation, methodology, evaluation, and discussion.

If you use or reference this code, please cite appropriately.