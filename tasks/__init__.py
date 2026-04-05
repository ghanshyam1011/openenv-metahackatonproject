"""Task loading utilities for the hallucination environment."""

from __future__ import annotations

from typing import Callable

from . import task1_easy, task2_medium, task3_hard


def _collect_samples(get_sample: Callable[[int], dict], total: int) -> list[dict]:
	"""Collect all samples exposed by a task module."""
	return [get_sample(i) for i in range(total)]


def load_samples(task_id: int, seed: int | None = None) -> list[dict]:
	"""Load samples for the requested task id.

	Args:
		task_id: Task difficulty id (1, 2, or 3).
		seed: Optional seed forwarded to task reset for reproducible shuffling.
	"""
	registry = {
		1: task1_easy,
		2: task2_medium,
		3: task3_hard,
	}

	if task_id not in registry:
		raise ValueError(f"Unknown task_id: {task_id}. Expected one of 1, 2, 3.")

	task_module = registry[task_id]
	task_module.reset(seed=seed)
	return _collect_samples(task_module.get_sample, task_module.total_samples())


__all__ = ["load_samples"]
