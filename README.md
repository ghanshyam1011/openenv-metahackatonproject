# OpenEnv — Hallucination Detection Environment

## 1. What This Environment Is and Why It Matters

OpenEnv is an open evaluation environment designed to benchmark AI agents on their ability to detect hallucinations in text. Hallucinations — false, fabricated, or misleading statements presented as fact — are one of the most critical failure modes of large language models in real-world deployment.

This environment provides a structured, reproducible way to measure how well an LLM agent can identify hallucinated content across three difficulty levels: easy factual myths, subtle medium-difficulty errors embedded in plausible academic language, and hard reasoning failures that require step-by-step analysis.

OpenEnv follows a standard gym-style loop: the environment gives the agent an observation (a text passage), the agent returns an action (its verdict), and the environment returns a reward score based on a task-specific grader.

---

## 2. What the Agent Does

The agent's job is **hallucination detection**. For each text passage it receives, it must:

- Decide whether the text contains a hallucination (`is_hallucination: true/false`)
- Express its confidence as a float between 0.0 and 1.0
- Provide a reason explaining its verdict, ideally naming specific errors

The agent interacts with the environment through two API calls: `POST /reset` to start a task, and `POST /step` to submit an action and receive the next observation and reward.

---

## 3. Observation Space and Action Space

**Observation** — what the agent receives each step:

| Field | Type | Description |
|---|---|---|
| `text` | string | The text passage to evaluate |
| `task_id` | integer | Which task is active (1, 2, or 3) |
| `sample_id` | integer | Index of the current sample |

**Action** — what the agent must return:

| Field | Type | Description |
|---|---|---|
| `is_hallucination` | boolean | `true` if the text contains a hallucination |
| `confidence` | float [0.0–1.0] | How confident the agent is |
| `reason` | string | Explanation, naming specific errors if any |

---

## 4. Task Descriptions

**Task 1 — Easy** (`easy_hallucination_detection`)

75 samples (50 hallucinated, 25 correct). Errors are common myths and widely debunked misconceptions visible to any model with general knowledge — e.g. "lightning never strikes twice" or "Napoleon was extremely short."

Difficulty: Easy

**Task 2 — Medium** (`medium_hallucination_detection`)

75 samples (30 hallucinated, 45 correct). Errors are embedded in longer, academic-sounding prose involving specific wrong entities, fabricated statistics, or overgeneralized claims that require careful reading.

Difficulty: Medium

**Task 3 — Hard** (`hard_hallucination_detection`)

20 manually crafted samples (12 hallucinated, 8 correct). These are confident, well-structured paragraphs containing subtle errors: wrong math shown step-by-step, fake citations with real author names, correct premises leading to wrong conclusions, and fabricated statistics. The agent must identify which specific step or claim is wrong.

Difficulty: Hard

---

## 5. How to Run Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-org/openenv.git
cd openenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```

The environment is now live at `http://localhost:7860`.

**Test with curl:**

```bash
# Reset to task 1
curl -X POST "http://localhost:7860/reset?task_id=1"

# Submit an action
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"is_hallucination": true, "confidence": 0.9, "reason": "lightning does strike the same place twice"}'

# Check state
curl "http://localhost:7860/state"
```

**Test with Python directly (no server needed):**

```python
import sys
sys.path.insert(0, ".")
from env import OpenEnv, Action

env = OpenEnv()
obs = env.reset(task_id=1)
print(obs)

action = Action(is_hallucination=True, confidence=0.85, reason="This is a myth about lightning")
next_obs, reward, done, info = env.step(action)
print(reward)
```

---

## 6. How to Run inference.py

`inference.py` runs a real LLM agent through all three tasks (10 samples each) and prints average scores.

**Step 1 — Set up your API key:**

```bash
cp .env.example .env
# Edit .env and fill in API_KEY, MODEL_NAME, API_BASE_URL
```

**Step 2 — Run:**

```bash
python inference.py
```

**Switching providers** — just change three lines in `.env`:

```bash
# Groq (fastest, recommended for free tier)
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
API_KEY=your_groq_key

# Gemini Flash (best quality free tier)
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
MODEL_NAME=gemini-2.5-flash
API_KEY=your_gemini_key

# Local Ollama (no key required)
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3.2
API_KEY=ollama
```

---

## 7. Baseline Scores

Indicative averages across 10 samples per task:

| Model | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Overall |
|---|---|---|---|---|
| Llama 3.3 70B (Groq) | 0.72 | 0.55 | 0.38 | 0.55 |
| Gemini 2.5 Flash | 0.78 | 0.61 | 0.44 | 0.61 |
| DeepSeek R1 | 0.80 | 0.65 | 0.52 | 0.66 |
| Mistral Small 3.2 | 0.65 | 0.48 | 0.30 | 0.48 |
| Random baseline | 0.25 | 0.20 | 0.15 | 0.20 |

---

## Grading Rubric

| Task | Correct Detection | Confidence Bonus | Reason Quality |
|---|---|---|---|
| Easy | +0.5 | +0.2 if confidence >= 0.7 | +0.3 if keyword matched |
| Medium | +0.4 | +0.2 if confidence >= 0.8 | +0.4 if wrong entity named |
| Hard | +0.3 | none | +0.4 step identified + 0.3 explains why |

---

## Project Structure

```
openenv/
├── env.py                  # Core environment (Observation, Action, Reward, OpenEnv)
├── api.py                  # FastAPI server wrapping the environment
├── inference.py            # LLM agent runner
├── openenv.yaml            # Environment spec
├── Dockerfile              # For HuggingFace Space deployment
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── data/
│   ├── easy_samples.json   # 75 easy samples
│   ├── medium_samples.json # 75 medium samples
│   └── hard_samples.json   # 20 hard samples
├── tasks/
│   ├── task1_easy.py
│   ├── task2_medium.py
│   └── task3_hard.py
└── graders/
    ├── grader1.py
    ├── grader2.py
    └── grader3.py
```
