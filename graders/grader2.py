"""
Grader 2 — Medium Task (HaluEval)

Uses openenv-core Rubric system.
Stricter than grader1 — agent must name the specific wrong entity
AND provide causal explanation (WHY, not just THAT).

Breakdown (max 1.0):
  correct_detection    0.35  — right verdict
  confidence           0.15  — calibrated confidence
  entity_identification 0.35 — reason names the specific wrong entity
  causal_explanation   0.15  — reason explains WHY (causal language)
"""

from openenv.core.rubrics import Rubric, WeightedSum

CAUSAL_SIGNALS = [
    "because", "since", "due to", "incorrect", "actually",
    "should be", "wrong because", "error is", "mistake",
    "in fact", "contradicts", "falsely", "inaccurate",
]


# ── Components ────────────────────────────────────────────────────────────────

class CorrectDetectionRubric(Rubric):
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        return 1.0 if action.is_hallucination == sample.get("label", False) else 0.0


class ConfidenceRubric(Rubric):
    """Medium task requires higher confidence threshold (0.75)."""
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        correct = action.is_hallucination == sample.get("label", False)
        if correct and action.confidence >= 0.75:
            return 1.0
        if correct and action.confidence >= 0.5:
            return 0.4
        if not correct and action.confidence <= 0.5:
            return 0.2
        return 0.0


class EntityIdentificationRubric(Rubric):
    """
    Agent must name the specific wrong entity from the ground truth.
    Partial credit if the agent identifies any part of the entity name.
    """
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        wrong_entity = sample.get("wrong_entity", "").lower().strip()
        if not wrong_entity:
            return 0.5  # No entity to check — partial credit
        reason_lower = action.reason.lower()

        # Full match
        if wrong_entity in reason_lower:
            return 1.0

        # Partial match — any word of the entity appears
        entity_words = [w for w in wrong_entity.split() if len(w) > 3]
        if entity_words and any(w in reason_lower for w in entity_words):
            return 0.4

        return 0.0


class CausalExplanationRubric(Rubric):
    """
    Agent must explain WHY the claim is wrong, not just THAT it is wrong.
    Looks for causal language signals.
    """
    def forward(self, action, observation) -> float:
        reason_lower = action.reason.lower()
        matches = sum(1 for sig in CAUSAL_SIGNALS if sig in reason_lower)
        if matches >= 2:
            return 1.0
        if matches == 1:
            return 0.5
        return 0.0


# ── Composed rubric ───────────────────────────────────────────────────────────

def build_rubric() -> WeightedSum:
    return WeightedSum(
        rubrics=[
            CorrectDetectionRubric(),
            ConfidenceRubric(),
            EntityIdentificationRubric(),
            CausalExplanationRubric(),
        ],
        weights=[0.35, 0.15, 0.35, 0.15],
    )


# ── grade() — called by env.py step() ────────────────────────────────────────

def grade(action, sample: dict) -> tuple[float, dict]:
    correct = action.is_hallucination == sample.get("label", False)
    reason_lower = action.reason.lower()
    wrong_entity = sample.get("wrong_entity", "").lower().strip()

    # Correct detection
    detect_score = 0.35 if correct else 0.0

    # Confidence
    if correct and action.confidence >= 0.75:
        conf_score = 0.15
    elif correct and action.confidence >= 0.5:
        conf_score = 0.06
    elif not correct and action.confidence <= 0.5:
        conf_score = 0.03
    else:
        conf_score = 0.0

    # Entity identification
    if not wrong_entity:
        entity_score = 0.175  # partial credit
    elif wrong_entity in reason_lower:
        entity_score = 0.35
    elif any(w in reason_lower for w in wrong_entity.split() if len(w) > 3):
        entity_score = 0.14  # 0.4 * 0.35
    else:
        entity_score = 0.0

    # Causal explanation
    matches = sum(1 for sig in CAUSAL_SIGNALS if sig in reason_lower)
    if matches >= 2:
        causal_score = 0.15
    elif matches == 1:
        causal_score = 0.075
    else:
        causal_score = 0.0

    breakdown = {
        "correct_detection":    round(detect_score, 4),
        "confidence":           round(conf_score, 4),
        "entity_identification": round(entity_score, 4),
        "causal_explanation":   round(causal_score, 4),
    }
    total = round(sum(breakdown.values()), 4)
    return total, breakdown