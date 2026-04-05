"""
Task 1 — Easy: Common Myths and Misconceptions

Primary source : HuggingFace -> truthful_qa (generation, validation split)
Fallback source: data/easy_samples.json (local)

What changes vs old file:
  - Loads real TruthfulQA dataset from HuggingFace when available
  - incorrect_answers from TruthfulQA become hallucinated samples (label=True)
  - best_answer from TruthfulQA becomes correct samples (label=False)
  - Falls back to local JSON if HF is unreachable
"""

import json
import os
import random

_samples = []

_BASE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FILE = os.path.join(_BASE, "..", "data", "easy_samples.json")


def _load_from_hf() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = []

    # Hallucinated: incorrect answers from TruthfulQA
    for i, row in enumerate(ds):
        if not row.get("incorrect_answers"):
            continue
        wrong = row["incorrect_answers"][0]
        source = row.get("source", "")
        keywords = [w for w in source.lower().replace("_", " ").split() if len(w) > 3]
        samples.append({
            "id": f"tqa_h_{i}",
            "text": f"Question: {row['question']}\nAnswer: {wrong}",
            "label": True,
            "keywords": keywords[:5],
            "source": "truthful_qa",
        })
        if sum(1 for s in samples if s["label"]) >= 50:
            break

    # Correct: best_answer from TruthfulQA
    correct_added = 0
    for i, row in enumerate(ds):
        if correct_added >= 25:
            break
        if not row.get("best_answer"):
            continue
        samples.append({
            "id": f"tqa_c_{i}",
            "text": f"Question: {row['question']}\nAnswer: {row['best_answer']}",
            "label": False,
            "keywords": [],
            "source": "truthful_qa",
        })
        correct_added += 1

    return samples


def _load_from_local() -> list[dict]:
    with open(_LOCAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def reset(seed=None):
    global _samples
    try:
        _samples = _load_from_hf()
        print(f"[Task 1] Loaded {len(_samples)} from HuggingFace (truthful_qa)")
    except Exception as e:
        _samples = _load_from_local()
        print(f"[Task 1] HF unavailable ({type(e).__name__}) — local fallback ({len(_samples)} samples)")
    if seed is not None:
        random.seed(seed)
        random.shuffle(_samples)


def get_sample(idx: int) -> dict:
    return _samples[idx]


def total_samples() -> int:
    return len(_samples)