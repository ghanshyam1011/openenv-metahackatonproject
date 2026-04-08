from fastapi import FastAPI, HTTPException, Query
from env import HallucinationEnv, HallucinationAction

app = FastAPI(
    title="OpenEnv — Hallucination Detection Environment",
    version="1.0.0",
)

env = HallucinationEnv()

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv is running."}

@app.post("/reset")
def reset(task_id: int = Query(default=1)):
    if task_id not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3.")
    obs = env.reset(task_id=task_id)
    return obs.model_dump()

@app.post("/step")
def step(action: HallucinationAction):
    try:
        obs = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }

@app.get("/state")
def state():
    return env.state.model_dump()
