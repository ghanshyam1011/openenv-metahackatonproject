"""
Grader 3 — Hard Task (Manually crafted reasoning errors)

Uses openenv-core LLMJudge for qualitative scoring.
This is the key differentiator — grader3 uses a real LLM to judge
whether the agent correctly identified the EXACT failing reasoning step
and explained WHY it fails.

Architecture:
  WeightedSum of:
    CorrectDetectionRubric  (0.25) — deterministic, no LLM
    LLMJudge                (0.75) — LLM scores step identification + explanation

LLMJudge prompt asks the judge model to score 0.0-1.0 based on:
  - Did the agent identify the exact failing step?  (0.0-0.5)
  - Did the agent explain WHY it fails?             (0.0-0.5)

Fallback: if LLM is unavailable, uses deterministic keyword scoring.
"""

import asyncio
import os
import re

from openenv.core.rubrics import LLMJudge, Rubric, WeightedSum

CAUSAL_SIGNALS = [
    "because", "since", "due to", "incorrect", "actually",
    "should be", "wrong because", "error occurs", "fails because",
    "mistake is", "the correct", "instead", "rather than",
]

LLM_JUDGE_PROMPT = """You are an expert evaluator scoring an AI agent's hallucination detection performance.

The agent was shown a text passage containing a reasoning error. 
Here is the GROUND TRUTH about what is wrong:
  Which step fails: {which_step}

The agent's response:
  Verdict: {"hallucinated" if is_hallucination else "not hallucinated"}
  Confidence: {confidence}
  Reason given: {reason}

Score the agent's reason on TWO criteria (each out of 0.5):

1. STEP IDENTIFICATION (0.0 to 0.5):
   - 0.5: Agent precisely identifies the exact failing step described in the ground truth
   - 0.3: Agent identifies the general area but not the specific step
   - 0.1: Agent mentions something related but misses the core failure
   - 0.0: Agent does not identify the failing step at all

2. EXPLANATION QUALITY (0.0 to 0.5):
   - 0.5: Agent explains clearly WHY the step is wrong (causal, specific reasoning)
   - 0.3: Agent hints at why but is vague or incomplete
   - 0.1: Agent says it is wrong without explaining why
   - 0.0: No explanation of why

Add both scores together. Return ONLY a single decimal number between 0.0 and 1.0.
Example valid responses: 0.8 or 0.35 or 1.0 or 0.0
Do not include any other text."""


# ── Deterministic components ──────────────────────────────────────────────────

class CorrectDetectionRubric(Rubric):
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        return 1.0 if action.is_hallucination == sample.get("label", False) else 0.0


class DeterministicReasoningRubric(Rubric):
    """
    Fallback rubric used when LLM judge is unavailable.
    Scores based on keyword overlap with which_step_is_wrong
    and presence of causal language.
    """
    def forward(self, action, observation) -> float:
        sample = observation.metadata.get("current_sample", {})
        which_step = sample.get("which_step_is_wrong", "").lower()
        reason_lower = action.reason.lower()

        # Step identification via word overlap
        step_words = [w for w in which_step.split() if len(w) > 4]
        if step_words:
            hits = sum(1 for w in step_words if w in reason_lower)
            step_score = min(hits / max(len(step_words), 1), 1.0) * 0.5
        else:
            step_score = 0.0

        # Causal explanation
        matches = sum(1 for sig in CAUSAL_SIGNALS if sig in reason_lower)
        causal_score = min(matches / 2, 1.0) * 0.5

        # Reasoning depth bonus
        depth_bonus = 0.1 if len(action.reason.split()) >= 20 else 0.0

        return round(min(step_score + causal_score + depth_bonus, 1.0), 4)


# ── LLM Judge setup ───────────────────────────────────────────────────────────

def _build_llm_judge():
    """
    Build the LLMJudge using openenv-core's OpenAIClient.
    Uses Groq by default (fastest free tier). Falls back gracefully.
    """
    api_key = os.getenv("API_KEY", "")
    model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

    if not api_key:
        return None

    try:
        # Parse endpoint and port from base_url
        # e.g. "https://api.groq.com/openai/v1" -> endpoint="https://api.groq.com", port=443
        from openenv.core import OpenAIClient

        # OpenAIClient expects endpoint + port separately
        # We use a custom subclass to support full base_url
        from openai import AsyncOpenAI

        class FullURLClient:
            """Thin wrapper that uses the full base_url directly."""
            def __init__(self, base_url, api_key, model):
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                self.model = model

            async def complete(self, prompt, **kwargs):
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20,
                )
                return resp.choices[0].message.content or "0.0"

        client = FullURLClient(base_url, api_key, model)

        judge = LLMJudge(
            prompt_template="{action}",  # We format manually in grade()
            client=client,
            score_pattern=r"(\d+\.?\d*)",
            default_score=0.3,
            normalize=True,
        )
        return judge, client

    except Exception as e:
        print(f"[Grader 3] LLM judge setup failed ({e}) — using deterministic fallback")
        return None


# ── grade() — called by env.py step() ────────────────────────────────────────

def grade(action, sample: dict) -> tuple[float, dict]:
    """
    Score the agent's action on a hard reasoning-error sample.

    Tries LLM-as-a-judge for the reasoning quality score.
    Falls back to deterministic keyword scoring if LLM unavailable.
    """
    correct = action.is_hallucination == sample.get("label", False)
    which_step = sample.get("which_step_is_wrong", "").lower()
    reason_lower = action.reason.lower()

    # ── 1. Correct detection (always deterministic) ───────────────────────────
    detect_score = 0.25 if correct else 0.0

    # ── 2. Reasoning quality (LLM judge or fallback) ──────────────────────────
    llm_result = _try_llm_judge(action, sample, which_step)

    if llm_result is not None:
        reasoning_score = llm_result * 0.75
        method = "llm_judge"
    else:
        # Deterministic fallback
        step_words = [w for w in which_step.split() if len(w) > 4]
        hits = sum(1 for w in step_words if w in reason_lower) if step_words else 0
        step_score = min(hits / max(len(step_words), 1), 1.0) * 0.35 if step_words else 0.0

        causal_matches = sum(1 for sig in CAUSAL_SIGNALS if sig in reason_lower)
        causal_score = min(causal_matches / 2, 1.0) * 0.20

        depth_score = 0.10 if len(action.reason.split()) >= 20 else 0.05 if len(action.reason.split()) >= 10 else 0.0

        reasoning_score = step_score + causal_score + depth_score
        method = "deterministic"

    breakdown = {
        "correct_detection": round(detect_score, 4),
        "reasoning_quality": round(reasoning_score, 4),
        "method": method,
    }
     total = round(min(detect_score + reasoning_score, 1.0), 4)
     total = max(0.0001, min(total, 0.9999))  # ← add this line
     return total, breakdown


def _try_llm_judge(action, sample, which_step) -> float | None:
    """
    Attempt to score reasoning quality using an LLM judge.
    Returns a float in [0,1] or None if unavailable.
    """
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        client = OpenAI(api_key=api_key, base_url=base_url)

        prompt = LLM_JUDGE_PROMPT.format(
            which_step=which_step or "not specified",
            is_hallucination=action.is_hallucination,
            confidence=action.confidence,
            reason=action.reason,
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content or "0"
        match = re.search(r"(\d+\.?\d*)", raw)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return None

    except Exception as e:
        print(f"[Grader 3] LLM judge call failed ({type(e).__name__}) — using deterministic")
        return None
