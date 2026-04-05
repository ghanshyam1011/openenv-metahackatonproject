from fastapi import FastAPI, HTTPException
from env import OpenEnv, Action, Observation, Reward
from typing import Optional

app = FastAPI(
    title="OpenEnv — Hallucination Detection Environment",
    version="1.0.0",
    description="An evaluation environment for testing AI agents on hallucination detection.",
)

env = OpenEnv()


@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv is running. Use /reset, /step, /state."}


@app.post("/reset")
def reset(task_id: int = 1):
    if task_id not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3.")
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    try:
        next_obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "next_observation": next_obs.model_dump() if next_obs else None,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()