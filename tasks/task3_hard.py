"""
Task 3 — Hard: Reasoning Errors (manually crafted)

Source: data/hard_samples.json (local only — always)

What changes vs old file:
  - No HF dataset — these are manually crafted reasoning errors that
    require human authoring to be subtle and non-trivial for frontier models
  - Added seed support for reproducible shuffling
  - Added source field to each sample for traceability
"""

import json
import os
import random

_samples = []

_BASE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FILE = os.path.join(_BASE, "..", "data", "hard_samples.json")


def reset(seed=None):
    """
    Task 3 always loads from local JSON.
    These are hand-crafted reasoning-error samples that do not exist
    in any public HuggingFace dataset.
    """
    global _samples
    with open(_LOCAL_FILE, encoding="utf-8") as f:
        _samples = json.load(f)
    # Add source field if missing
    for s in _samples:
        s.setdefault("source", "manual")
    if seed is not None:
        random.seed(seed)
        random.shuffle(_samples)
    print(f"[Task 3] Loaded {len(_samples)} manually crafted reasoning-error samples")


def get_sample(idx: int) -> dict:
    return _samples[idx]


def total_samples() -> int:
    return len(_samples)