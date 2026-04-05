import json
import os

_samples = []
_index = 0


def reset():
    global _samples, _index
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hard_samples.json")
    with open(data_path, "r") as f:
        _samples = json.load(f)
    _index = 0


def get_sample(idx: int) -> dict:
    return _samples[idx]


def total_samples() -> int:
    return len(_samples)
