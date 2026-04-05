from pydantic import BaseModel, Field
from typing import Optional
import importlib
import os


# ─── Pydantic Models ────────────────────────────────────────────────────────

class Observation(BaseModel):
    text: str
    task_id: int
    sample_id: int


class Action(BaseModel):
    is_hallucination: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict


# ─── Task + Grader Registry ──────────────────────────────────────────────────

TASKS = {
    1: ("tasks.task1_easy",   "graders.grader1"),
    2: ("tasks.task2_medium", "graders.grader2"),
    3: ("tasks.task3_hard",   "graders.grader3"),
}


# ─── Environment ─────────────────────────────────────────────────────────────

class OpenEnv:
    def __init__(self):
        self._task_id: int = 1
        self._sample_id: int = 0
        self._task_module = None
        self._grader_module = None
        self._done: bool = False

    def reset(self, task_id: int = 1) -> Observation:
        """Load the first sample from the chosen task and return an Observation."""
        self._task_id = task_id
        self._sample_id = 0
        self._done = False

        task_mod_path, grader_mod_path = TASKS[task_id]
        self._task_module = importlib.import_module(task_mod_path)
        self._grader_module = importlib.import_module(grader_mod_path)

        self._task_module.reset()
        sample = self._task_module.get_sample(self._sample_id)

        return Observation(
            text=sample["text"],
            task_id=self._task_id,
            sample_id=self._sample_id,
        )

    def step(self, action: Action):
        """
        Receive an Action, call the grader, advance to the next sample.
        Returns: (next_observation | None, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Grade the action
        sample = self._task_module.get_sample(self._sample_id)
        score, breakdown = self._grader_module.grade(action, sample)
        reward = Reward(score=score, breakdown=breakdown)

        # Advance
        self._sample_id += 1
        total = self._task_module.total_samples()

        if self._sample_id >= total:
            self._done = True
            return None, reward, True, {"info": "All samples completed."}

        next_sample = self._task_module.get_sample(self._sample_id)
        next_obs = Observation(
            text=next_sample["text"],
            task_id=self._task_id,
            sample_id=self._sample_id,
        )
        return next_obs, reward, False, {}

    def state(self) -> dict:
        """Return current task and sample index."""
        return {
            "task_id": self._task_id,
            "sample_id": self._sample_id,
            "done": self._done,
        }
