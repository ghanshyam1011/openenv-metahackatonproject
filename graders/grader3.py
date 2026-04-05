"""
Grader 3 — Hard task scoring (hardest, most specific)

Scoring breakdown:
  +0.3  correct hallucination detection
  +0.4  reason identifies the exact step where reasoning fails (which_step_is_wrong)
  +0.3  reason explains WHY it is wrong (contains explanation keywords: because/since/due/incorrect/wrong because)
"""

from env import Action


def grade(action: Action, sample: dict) -> tuple[float, dict]:
    ground_truth_label: bool = sample["label"]
    which_step: str = sample.get("which_step_is_wrong", "").lower()

    breakdown = {
        "correct_detection": 0.0,
        "step_identification": 0.0,
        "explains_why": 0.0,
    }

    correct = action.is_hallucination == ground_truth_label

    if correct:
        breakdown["correct_detection"] = 0.3

    reason_lower = action.reason.lower()

    # Check if reason identifies the failing step
    if which_step and which_step in reason_lower:
        breakdown["step_identification"] = 0.4

    # Check if reason explains WHY (not just THAT) it is wrong
    explanation_signals = [
        "because", "since", "due to", "this is wrong", "incorrect because",
        "error occurs", "mistake is", "fails because", "should be", "actually"
    ]
    if any(signal in reason_lower for signal in explanation_signals):
        breakdown["explains_why"] = 0.3

    score = round(sum(breakdown.values()), 4)
    return score, breakdown
