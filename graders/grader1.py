"""
Grader 1 — Easy Task (TruthfulQA)

Uses openenv-core Rubric system.
Scoring is deterministic (no LLM needed for easy task — fast and reliable).

Breakdown (max 1.0):
  correct_detection   0.40  — right answer: hallucinated or not
  confidence          0.20  — confident when correct, uncertain when wrong
  keyword_specificity 0.25  — reason mentions a specific keyword from the sample
  reasoning_depth     0.15  — reason is detailed enough (8+ words)
"""

from openenv.core.rubrics import Rubric, WeightedSum


# ── Individual rubric components ──────────────────────────────────────────────

class CorrectDetectionRubric(Rubric):
    """Full credit if the agent correctly labels hallucinated vs correct."""
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        if not sample:
            return 0.0
        return 1.0 if action.is_hallucination == sample.get("label", False) else 0.0


class ConfidenceRubric(Rubric):
    """
    Rewards calibrated confidence.
    - Correct + confident (>=0.7) → 1.0
    - Correct + uncertain (<0.7)  → 0.3  (partial credit)
    - Wrong + overconfident (>0.6)→ 0.0  (penalised)
    - Wrong + uncertain (<=0.6)   → 0.2  (appropriate uncertainty rewarded)
    """
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        if not sample:
            return 0.0
        correct = action.is_hallucination == sample.get("label", False)
        if correct and action.confidence >= 0.7:
            return 1.0
        if correct and action.confidence < 0.7:
            return 0.3
        if not correct and action.confidence > 0.6:
            return 0.0
        if not correct and action.confidence <= 0.6:
            return 0.2
        return 0.0


class KeywordSpecificityRubric(Rubric):
    """
    Rewards mentioning a specific keyword from the ground truth.
    Prevents vague generic answers from scoring high.
    """
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        if not sample:
            return 0.0
        keywords = [k.lower() for k in sample.get("keywords", [])]
        if not keywords:
            return 0.5  # No keywords to check — give partial credit
        reason_lower = action.reason.lower()
        return 1.0 if any(k in reason_lower for k in keywords) else 0.0


class ReasoningDepthRubric(Rubric):
    """
    Rewards sufficiently detailed explanations.
    One-word or very short reasons score 0 — forces agents to explain.
    """
    def forward(self, action, observation) -> float:
        words = action.reason.split()
        if len(words) >= 15:
            return 1.0
        if len(words) >= 8:
            return 0.6
        return 0.0


# ── Composed rubric ───────────────────────────────────────────────────────────

def build_rubric() -> WeightedSum:
    return WeightedSum(
        rubrics=[
            CorrectDetectionRubric(),
            ConfidenceRubric(),
            KeywordSpecificityRubric(),
            ReasoningDepthRubric(),
        ],
        weights=[0.40, 0.20, 0.25, 0.15],
    )


# ── grade() — called by env.py step() ────────────────────────────────────────

def grade(action, sample: dict) -> tuple[float, dict]:
    """
    Score the agent's action against the ground truth sample.
    Returns (total_score, breakdown_dict).
    Called synchronously from env.py step().
    """
    correct = action.is_hallucination == sample.get("label", False)
    reason_lower = action.reason.lower()
    keywords = [k.lower() for k in sample.get("keywords", [])]

    # Correct detection
    correct_score = 0.40 if correct else 0.0

    # Confidence calibration
    if correct and action.confidence >= 0.7:
        conf_score = 0.20
    elif correct and action.confidence < 0.7:
        conf_score = 0.06  # 0.3 * 0.20
    elif not correct and action.confidence <= 0.6:
        conf_score = 0.04  # 0.2 * 0.20
    else:
        conf_score = 0.0

    # Keyword specificity
    if keywords:
        kw_score = 0.25 if any(k in reason_lower for k in keywords) else 0.0
    else:
        kw_score = 0.125  # partial credit when no keywords

    # Reasoning depth
    word_count = len(action.reason.split())
    if word_count >= 15:
        depth_score = 0.15
    elif word_count >= 8:
        depth_score = 0.09  # 0.6 * 0.15
    else:
        depth_score = 0.0

    breakdown = {
        "correct_detection":   round(correct_score, 4),
        "confidence":          round(conf_score, 4),
        "keyword_specificity": round(kw_score, 4),
        "reasoning_depth":     round(depth_score, 4),
    }
   total = round(sum(breakdown.values()), 4)
  total = max(0.0001, min(total, 0.9999))  # ← add this
  return total, breakdown
