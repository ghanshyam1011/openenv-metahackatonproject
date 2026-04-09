"""
Hallucination Detection Environment — built on openenv-core.

A reinforcement learning environment where an agent reads text passages
and must detect whether they contain hallucinations (false claims presented
as fact). Supports 3 difficulty levels with progressively harder tasks.
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

from openenv.core import Action, Environment, Observation, State
from pydantic import Field
from tasks import load_samples


# ── Observation ───────────────────────────────────────────────────────────────

class HallucinationObservation(Observation):
    """What the agent sees at each step."""

    text: str = Field(
        default="",
        description="Text passage the agent must evaluate for hallucinations.",
    )
    task_id: int = Field(
        default=1,
        description="Active task difficulty: 1=easy, 2=medium, 3=hard.",
    )
    sample_id: int = Field(
        default=0,
        description="Index of the current sample within the task (0-based).",
    )
    total_samples: int = Field(
        default=0,
        description="Total number of samples in this task episode.",
    )
    task_name: str = Field(
        default="",
        description="Human-readable name of the current task.",
    )


# ── Action ────────────────────────────────────────────────────────────────────

class HallucinationAction(Action):
    """What the agent returns at each step."""

    is_hallucination: bool = Field(
        description=(
            "Set to true if the text contains a hallucination — "
            "a factual error, fabricated statistic, fake citation, "
            "or false claim presented as truth."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the decision, from 0.0 (uncertain) to 1.0 (certain). "
            "A well-calibrated agent should use high confidence only when the "
            "error is clear and specific."
        ),
    )
    reason: str = Field(
        description=(
            "Explanation of the decision. For hallucinations: name the specific "
            "false claim, identify which part is wrong, and explain WHY it is wrong — "
            "not just that it is wrong. For correct text: explain why it appears accurate."
        )
    )


# ── State ─────────────────────────────────────────────────────────────────────

class HallucinationState(State):
    """Internal environment state — not visible to the agent."""

    task_id: int = Field(default=1)
    sample_id: int = Field(default=0)
    samples: list[dict] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


# ── Task Registry ─────────────────────────────────────────────────────────────

TASK_CONFIG = {
    1: {
        "name": "Easy — Common Myths",
        "file": "data/easy_samples.json",
        "description": "Common misconceptions and myths. Errors are clear and detectable.",
    },
    2: {
        "name": "Medium — Subtle Claims",
        "file": "data/medium_samples.json",
        "description": "Academic-sounding passages with subtle hallucinations. Requires careful reading.",
    },
    3: {
        "name": "Hard — Reasoning Errors",
        "file": "data/hard_samples.json",
        "description": "Confident-sounding passages with hidden reasoning errors. Requires identifying exactly which step fails.",
    },
}


# ── Reward Logic ──────────────────────────────────────────────────────────────

def compute_reward(action: HallucinationAction, sample: dict, task_id: int) -> tuple[float, dict]:
    """
    Compute a dense reward signal for the agent's action.

    The reward is designed to encourage:
    - Correct hallucination detection (binary)
    - Well-calibrated confidence (not just being right, but being right confidently)
    - Specific, detailed reasoning (not vague or gameable)

    Returns (score, breakdown) where score is in [0.0, 1.0].
    """
    label: bool = sample["label"]
    reason_lower = action.reason.lower()
    correct = action.is_hallucination == label

    breakdown: dict[str, float] = {}

    # ── Task 1: Easy ──────────────────────────────────────────────────────────
    if task_id == 1:
        # Correct detection
        breakdown["correct_detection"] = 0.4 if correct else 0.0

        # Confidence calibration — reward confident correct answers
        if correct and action.confidence >= 0.7:
            breakdown["confidence_calibration"] = 0.2
        elif not correct and action.confidence <= 0.5:
            # Reward appropriate uncertainty when wrong (partial credit)
            breakdown["confidence_calibration"] = 0.05
        else:
            breakdown["confidence_calibration"] = 0.0

        # Reasoning quality — does the reason mention a specific error?
        keywords = [k.lower() for k in sample.get("keywords", [])]
        keyword_hit = any(k in reason_lower for k in keywords) if keywords else False
        breakdown["keyword_specificity"] = 0.25 if keyword_hit else 0.0

        # Reasoning depth — penalise one-word reasons
        breakdown["reasoning_depth"] = 0.15 if len(action.reason.split()) >= 8 else 0.0

    # ── Task 2: Medium ────────────────────────────────────────────────────────
    elif task_id == 2:
        breakdown["correct_detection"] = 0.35 if correct else 0.0

        if correct and action.confidence >= 0.75:
            breakdown["confidence_calibration"] = 0.15
        else:
            breakdown["confidence_calibration"] = 0.0

        # Must name the specific wrong entity
        wrong_entity = sample.get("wrong_entity", "").lower()
        entity_named = wrong_entity and wrong_entity in reason_lower
        breakdown["entity_identification"] = 0.35 if entity_named else 0.0

        # Must explain why (not just that) — look for causal language
        causal_signals = ["because", "since", "due to", "incorrect", "actually",
                         "should be", "wrong because", "error is", "mistake"]
        explains_why = any(sig in reason_lower for sig in causal_signals)
        breakdown["causal_explanation"] = 0.15 if explains_why else 0.0

    # ── Task 3: Hard ──────────────────────────────────────────────────────────
    elif task_id == 3:
        breakdown["correct_detection"] = 0.25 if correct else 0.0

        if correct and action.confidence >= 0.8:
            breakdown["confidence_calibration"] = 0.1
        else:
            breakdown["confidence_calibration"] = 0.0

        # Must identify the EXACT failing step
        which_step = sample.get("which_step_is_wrong", "").lower()
        # Check at least 4 consecutive words from the step description appear in reason
        if which_step:
            step_words = which_step.split()
            hits = sum(1 for w in step_words if w in reason_lower)
            step_ratio = hits / max(len(step_words), 1)
            breakdown["step_identification"] = round(0.35 * min(step_ratio * 2, 1.0), 4)
        else:
            breakdown["step_identification"] = 0.0

        # Must explain WHY the step is wrong (causal language + substance)
        causal_signals = ["because", "since", "due to", "incorrect", "actually",
                         "should be", "wrong because", "error occurs", "fails because",
                         "mistake is", "the correct"]
        explains_why = any(sig in reason_lower for sig in causal_signals)
        breakdown["explains_why"] = 0.2 if explains_why else 0.0

        # Penalise vague reasons
        breakdown["reasoning_depth"] = 0.1 if len(action.reason.split()) >= 15 else 0.0

    total = round(sum(breakdown.values()), 4)
    total = max(0.0001, min(total, 0.9999))
    return total, breakdown


# ── Environment ───────────────────────────────────────────────────────────────

class HallucinationEnv(Environment):
    """
    Hallucination Detection RL Environment.

    An agent must read text passages and determine whether they contain
    hallucinations. The reward signal is designed to be dense and meaningful —
    it rewards not just correct detection but also confidence calibration,
    specific reasoning, and causal explanation.

    This environment is suitable for post-training LLMs using GRPO or PPO.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = HallucinationState()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_samples(self, task_id: int, seed=None) -> list[dict]:
        """
        Load samples for a given task.
        Tries HuggingFace datasets first, falls back to local JSON.
        """
        return load_samples(task_id, seed=seed)

    def _make_observation(self, done: bool = False, reward: float | None = None) -> HallucinationObservation:
        s = self._state
        if done or s.sample_id >= len(s.samples):
            return HallucinationObservation(
                text="Episode complete. Call reset() to start a new episode.",
                task_id=s.task_id,
                sample_id=s.sample_id,
                total_samples=len(s.samples),
                task_name=TASK_CONFIG[s.task_id]["name"],
                done=True,
                reward=reward,
            )
        sample = s.samples[s.sample_id]
        return HallucinationObservation(
            text=sample["text"],
            task_id=s.task_id,
            sample_id=s.sample_id,
            total_samples=len(s.samples),
            task_name=TASK_CONFIG[s.task_id]["name"],
            done=False,
            reward=reward,
        )

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs: Any,
    ) -> HallucinationObservation:
        """
        Reset the environment and return the first observation.

        Args:
            task_id: Which task to run. 1=easy, 2=medium, 3=hard.
            seed: Optional random seed for reproducibility.
        """
        self._reset_rubric()
        if task_id not in TASK_CONFIG:
            raise ValueError(f"task_id must be 1, 2, or 3. Got: {task_id}")

        samples = self._load_samples(task_id, seed=seed)
        if seed is not None:
            random.seed(seed)
            random.shuffle(samples)

        self._state = HallucinationState(
            task_id=task_id,
            sample_id=0,
            samples=samples,
            scores=[],
            episode_id=episode_id,
        )
        return self._make_observation()

    def step(
        self,
        action: HallucinationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HallucinationObservation:
        """
        Submit the agent's decision and receive the next observation + reward.

        The reward is a dense signal in [0.0, 1.0] that scores:
        - Correct hallucination detection
        - Confidence calibration
        - Specificity of the reason (entity named, step identified)
        - Causal explanation quality (WHY it's wrong, not just THAT it's wrong)
        """
        s = self._state
        if s.sample_id >= len(s.samples):
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_sample = s.samples[s.sample_id]
        score, breakdown = compute_reward(action, current_sample, s.task_id)

        s.scores.append(score)
        s.sample_id += 1
        s.step_count += 1

        done = s.sample_id >= len(s.samples)
        obs = self._make_observation(done=done, reward=score)
        obs.metadata["breakdown"] = breakdown
        obs.metadata["running_avg"] = round(sum(s.scores) / len(s.scores), 4)
        return obs

    @property
    def state(self) -> HallucinationState:
        return self._state
