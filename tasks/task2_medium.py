"""
Task 2 — Medium: Subtle Hallucinations in Academic-style Text

Primary source : HuggingFace -> pminervini/HaluEval (general split)
Fallback source: data/medium_samples.json (local)

What changes vs old file:
  - Loads real HaluEval dataset from HuggingFace when available
  - hallucinated_answer field becomes hallucinated samples (label=True)
  - right_answer field becomes correct samples (label=False)
  - Extracts wrong_entity by comparing hallucinated vs correct answer
  - Falls back to local JSON if HF is unreachable
"""

import json
import os
import random

_samples = []

_BASE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FILE = os.path.join(_BASE, "..", "data", "medium_samples.json")


def _load_from_hf() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "general", split="data")
    samples = []

    # Hallucinated: use hallucinated_answer field
    for i, row in enumerate(ds):
        if not row.get("hallucinated_answer"):
            continue
        wrong = row["hallucinated_answer"]
        right = row.get("right_answer", "")
        # Find tokens unique to the wrong answer to use as wrong_entity hint
        wrong_words = set(wrong.lower().split())
        right_words = set(right.lower().split())
        unique_wrong = [w for w in wrong_words - right_words if len(w) > 4]
        wrong_entity = unique_wrong[0] if unique_wrong else wrong[:30].lower()

        knowledge = row.get("knowledge", "")
        text = (
            f"{knowledge}\n\nQuestion: {row['question']}\nAnswer: {wrong}"
            if knowledge else
            f"Question: {row['question']}\nAnswer: {wrong}"
        )
        samples.append({
            "id": f"halueval_h_{i}",
            "text": text.strip(),
            "label": True,
            "wrong_entity": wrong_entity,
            "source": "halueval",
        })
        if sum(1 for s in samples if s["label"]) >= 30:
            break

    # Correct: use right_answer field
    correct_added = 0
    for i, row in enumerate(ds):
        if correct_added >= 45:
            break
        if not row.get("right_answer"):
            continue
        knowledge = row.get("knowledge", "")
        text = (
            f"{knowledge}\n\nQuestion: {row['question']}\nAnswer: {row['right_answer']}"
            if knowledge else
            f"Question: {row['question']}\nAnswer: {row['right_answer']}"
        )
        samples.append({
            "id": f"halueval_c_{i}",
            "text": text.strip(),
            "label": False,
            "wrong_entity": "",
            "source": "halueval",
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
        print(f"[Task 2] Loaded {len(_samples)} from HuggingFace (HaluEval)")
    except Exception as e:
        _samples = _load_from_local()
        print(f"[Task 2] HF unavailable ({type(e).__name__}) — local fallback ({len(_samples)} samples)")
    if seed is not None:
        random.seed(seed)
        random.shuffle(_samples)


def get_sample(idx: int) -> dict:
    return _samples[idx]


def total_samples() -> int:
    return len(_samples)