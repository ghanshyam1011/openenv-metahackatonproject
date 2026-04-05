"""
inference.py — Run an LLM agent through the Hallucination Detection Environment.

Uses openenv-core's SyncEnvClient to connect to a running environment server
and runs 10 samples per task, printing scores as it goes.

Usage:
    # Option 1: against your local server
    python inference.py --base-url http://localhost:7860

    # Option 2: against your live HF Space
    python inference.py --base-url https://YOUR_HF_USERNAME-openenv.hf.space

Environment variables (.env file):
    API_KEY       — Groq / OpenAI / Gemini key
    MODEL_NAME    — e.g. llama-3.3-70b-versatile
    API_BASE_URL  — LLM provider base URL
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from openenv import SyncEnvClient

load_dotenv()

LLM_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
API_KEY      = os.getenv("API_KEY", "")

SAMPLES_PER_TASK = 10

SYSTEM_PROMPT = """You are a hallucination detection expert.

Read the text carefully. Determine whether it contains a hallucination — 
meaning a factual error, fabricated statistic, invented citation, flawed 
reasoning step, or false claim presented as truth.

Respond ONLY with a valid JSON object:
{
  "is_hallucination": true or false,
  "confidence": float between 0.0 and 1.0,
  "reason": "Specific explanation. If hallucinated: name the exact false claim, identify which part is wrong, and explain WHY it is wrong. If correct: explain what makes it accurate."
}

No markdown. No extra text. Only the JSON object."""


def call_llm(client: OpenAI, text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Evaluate this text:\n\n{text}"},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(env_url: str, llm_client: OpenAI, task_id: int) -> list[float]:
    task_names = {1: "Easy (Common Myths)", 2: "Medium (Subtle Claims)", 3: "Hard (Reasoning Errors)"}
    print(f"\nTask {task_id} — {task_names[task_id]}")
    print("-" * 50)

    scores = []

    with SyncEnvClient(base_url=env_url) as client:
        result = client.reset(task_id=task_id)

        for i in range(SAMPLES_PER_TASK):
            text = result.observation.text
            print(f"  [{i+1}/{SAMPLES_PER_TASK}] {text[:65]}...")

            try:
                parsed = call_llm(llm_client, text)
                action_data = {
                    "is_hallucination": bool(parsed["is_hallucination"]),
                    "confidence": float(parsed["confidence"]),
                    "reason": str(parsed["reason"]),
                }
            except Exception as e:
                print(f"    LLM error: {e} — using default action")
                action_data = {"is_hallucination": False, "confidence": 0.5, "reason": "parse error"}

            result = client.step(action_data)
            score = result.observation.reward or 0.0
            breakdown = result.observation.metadata.get("breakdown", {})
            scores.append(score)
            print(f"    score={score:.3f}  breakdown={breakdown}")

            if result.observation.done:
                break

            time.sleep(0.3)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  Average: {avg:.3f}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:7860")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set API_KEY in your .env file.")
        return

    llm_client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

    print(f"\nOpenEnv Inference Runner")
    print(f"Environment : {args.base_url}")
    print(f"LLM Model   : {MODEL_NAME}")
    print(f"Samples     : {SAMPLES_PER_TASK} per task")
    print("=" * 50)

    start = time.time()
    all_scores = {}

    for task_id in [1, 2, 3]:
        all_scores[task_id] = run_task(args.base_url, llm_client, task_id)

    elapsed = time.time() - start

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for task_id, scores in all_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        bar = "█" * int(avg * 20)
        print(f"  Task {task_id}: {avg:.3f}  {bar}")

    overall = sum(
        sum(s) / len(s) for s in all_scores.values() if s
    ) / len(all_scores)
    print(f"\n  Overall: {overall:.3f}")
    print(f"  Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()