"""
inference.py — Agent for Hallucination Detection Environment.
Uses the hackathon-provided LiteLLM proxy via API_BASE_URL + API_KEY.
Prints [START]/[STEP]/[END] structured output blocks for the validator.
"""

from __future__ import annotations
import os
import json
import openai
from env import HallucinationEnv, HallucinationAction

# ── LiteLLM Proxy Client (injected by hackathon) ─────────────────────────────

client = openai.OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)

MODEL = os.environ.get("MODEL_NAME", "gpt-4o")

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert hallucination detector. You will be given a text passage.
Determine if it contains a hallucination — a factual error, fabricated statistic,
fake citation, or false claim presented as truth.

Respond ONLY with valid JSON. No markdown, no extra text. Format:
{
  "is_hallucination": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "Name the specific false claim and explain WHY it is wrong, not just that it is wrong."
}"""

TASK_HINTS = {
    1: "Easy task — common myths. Look for clear factual errors. Be confident if you spot one.",
    2: (
        "Medium task — subtle errors. Find the specific wrong entity (name, number, date). "
        "Your reason MUST name it and use words like 'because', 'incorrect', 'should be', or 'actually'."
    ),
    3: (
        "Hard task — reasoning errors. Find the exact failing step in the logic or math. "
        "Your reason MUST say which step fails and WHY, using 'because', 'fails because', "
        "'the correct', or 'mistake is'."
    ),
}

# ── API Caller ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def get_action(text: str, task_id: int) -> HallucinationAction:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{TASK_HINTS[task_id]}\n\nText:\n{text}"},
        ],
    )
    data = _parse_json(response.choices[0].message.content)
    return HallucinationAction(
        is_hallucination=bool(data["is_hallucination"]),
        confidence=float(data["confidence"]),
        reason=str(data["reason"]),
    )

# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_task(task_id: int) -> None:
    env = HallucinationEnv()
    obs = env.reset(task_id=task_id)
    task_name = obs.task_name.replace(" ", "_").replace("—", "-")

    print(f"[START] task={task_name}", flush=True)

    step_num = 0
    total_reward = 0.0

    while not obs.done:
        try:
            action = get_action(obs.text, task_id)
        except Exception as e:
            print(f"[WARN] API call failed: {e}", flush=True)
            action = HallucinationAction(
                is_hallucination=False,
                confidence=0.4,
                reason=f"Could not determine hallucination status because an error occurred: {e}",
            )

        obs = env.step(action)
        step_num += 1
        reward = obs.reward or 0.0
        total_reward += reward
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    final_score = round(total_reward / max(step_num, 1), 4)
    print(f"[END] task={task_name} score={final_score} steps={step_num}", flush=True)


if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        run_task(task_id)
