"""
Grader 2 — Medium task scoring (stricter than grader 1)

Scoring breakdown:
  +0.4  correct hallucination detection
  +0.2  confidence >= 0.8 when correct
  +0.4  reason mentions the specific wrong entity from ground truth
"""

from env import Action


def grade(action: Action, sample: dict) -> tuple[float, dict]:
    ground_truth_label: bool = sample["label"]
    wrong_entity: str = sample.get("wrong_entity", "").lower()

    breakdown = {
        "correct_detection": 0.0,
        "confidence_bonus": 0.0,
        "entity_match": 0.0,
    }

    correct = action.is_hallucination == ground_truth_label

    if correct:
        breakdown["correct_detection"] = 0.4
        if action.confidence >= 0.8:
            breakdown["confidence_bonus"] = 0.2

    reason_lower = action.reason.lower()
    if wrong_entity and wrong_entity in reason_lower:
        breakdown["entity_match"] = 0.4

    score = round(sum(breakdown.values()), 4)
    return score, breakdown
