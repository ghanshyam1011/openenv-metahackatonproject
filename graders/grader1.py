"""
Grader 1 — Easy task scoring

Scoring breakdown:
  +0.5  correct hallucination detection
  +0.2  confidence >= 0.7 when correct
  +0.3  reason mentions at least one keyword from ground truth
"""

from env import Action


def grade(action: Action, sample: dict) -> tuple[float, dict]:
    ground_truth_label: bool = sample["label"]
    keywords: list[str] = [kw.lower() for kw in sample.get("keywords", [])]

    breakdown = {
        "correct_detection": 0.0,
        "confidence_bonus": 0.0,
        "keyword_match": 0.0,
    }

    correct = action.is_hallucination == ground_truth_label

    if correct:
        breakdown["correct_detection"] = 0.5
        if action.confidence >= 0.7:
            breakdown["confidence_bonus"] = 0.2

    reason_lower = action.reason.lower()
    if any(kw in reason_lower for kw in keywords):
        breakdown["keyword_match"] = 0.3

    score = round(sum(breakdown.values()), 4)
    return score, breakdown
