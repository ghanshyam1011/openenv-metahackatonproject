"""
inference.py — Run an LLM agent through the OpenEnv hallucination detection environment.

Environment variables required:
  API_BASE_URL  — e.g. https://api.groq.com/openai/v1
  MODEL_NAME    — e.g. llama-3.3-70b-versatile
  API_KEY       — your provider's API key
  HF_TOKEN      — HuggingFace token (used if API_BASE_URL is HF Inference)

Usage:
  python inference.py

The script runs 10 samples per task (tasks 1, 2, 3) and prints average scores.
Total runtime target: under 20 minutes.
"""

import os
import json
import time
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file if present
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
API_KEY      = os.getenv("API_KEY",      "")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

SAMPLES_PER_TASK = 10
TASKS = [1, 2, 3]

SYSTEM_PROMPT = """You are a hallucination detection expert. 
You will be given a piece of text. Your job is to determine whether it contains a hallucination — 
meaning a factual error, fabricated citation, wrong statistic, or false claim presented as truth.

Respond ONLY with a valid JSON object in this exact format:
{
  "is_hallucination": true or false,
  "confidence": a float between 0.0 and 1.0,
  "reason": "a clear explanation of your decision, naming specific errors if any"
}

Do not include any other text outside the JSON object."""


def build_user_prompt(text: str) -> str:
    return f"Evaluate the following text for hallucinations:\n\n{text}"


def call_llm(client: OpenAI, text: str) -> dict:
    """Call the LLM and parse its response into an Action dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(text)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return {
        "is_hallucination": bool(parsed["is_hallucination"]),
        "confidence":       float(parsed["confidence"]),
        "reason":           str(parsed["reason"]),
    }


def run_task(env, client: OpenAI, task_id: int, n_samples: int) -> list[float]:
    """Run n_samples through a task and return the list of scores."""
    from env import Action

    obs_data = env.reset(task_id=task_id)
    scores = []

    for i in range(n_samples):
        text = obs_data["text"] if isinstance(obs_data, dict) else obs_data.text

        print(f"  [Task {task_id}] Sample {i+1}/{n_samples} — ", end="", flush=True)

        try:
            action_data = call_llm(client, text)
            action = Action(**action_data)
        except Exception as e:
            print(f"LLM error: {e} — skipping")
            # Advance with a default action to keep the episode going
            action = Action(is_hallucination=False, confidence=0.5, reason="error parsing response")

        result = env.step(action)
        _, reward, done, _ = result

        score = reward.score if hasattr(reward, "score") else reward["score"]
        scores.append(score)
        print(f"score={score:.3f}")

        if done or i == n_samples - 1:
            break

        obs_data = result[0]
        time.sleep(0.3)  # polite rate limiting

    return scores


def main():
    if not API_KEY and not HF_TOKEN:
        print("ERROR: Set API_KEY (or HF_TOKEN) environment variable.")
        sys.exit(1)

    key = API_KEY or HF_TOKEN
    client = OpenAI(api_key=key, base_url=API_BASE_URL)

    # Import env here so the module path works
    sys.path.insert(0, os.path.dirname(__file__))
    from env import OpenEnv

    env = OpenEnv()

    print(f"\nOpenEnv Inference Runner")
    print(f"Model     : {MODEL_NAME}")
    print(f"API base  : {API_BASE_URL}")
    print(f"Samples   : {SAMPLES_PER_TASK} per task\n")
    print("=" * 50)

    all_results = {}
    start_time = time.time()

    for task_id in TASKS:
        task_names = {1: "Easy (TruthfulQA)", 2: "Medium (HaluEval)", 3: "Hard (Manual)"}
        print(f"\nTask {task_id} — {task_names[task_id]}")
        print("-" * 40)

        scores = run_task(env, client, task_id, SAMPLES_PER_TASK)
        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task_id] = {"scores": scores, "average": avg}
        print(f"  Average score: {avg:.3f}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for task_id, result in all_results.items():
        bar = "█" * int(result["average"] * 20)
        print(f"  Task {task_id}: {result['average']:.3f}  {bar}")

    overall = sum(r["average"] for r in all_results.values()) / len(all_results)
    print(f"\n  Overall average : {overall:.3f}")
    print(f"  Total runtime   : {elapsed:.1f}s")

    if elapsed > 1200:
        print("\n  WARNING: Runtime exceeded 20 minutes target.")


if __name__ == "__main__":
    main()
