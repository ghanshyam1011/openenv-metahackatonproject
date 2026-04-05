"""
app.py — OpenEnv-powered FastAPI server with built-in /web UI.

This replaces the old api.py. Instead of hand-rolling endpoints,
we use openenv-core's create_web_interface_app which gives us:
  - POST /reset
  - POST /step
  - GET  /state
  - GET  /health
  - GET  /metadata
  - GET  /schema
  - GET  /web      ← interactive UI for judges
  - GET  /docs     ← Swagger UI
  - WebSocket /ws  ← for RL training loops
"""

from env import HallucinationAction, HallucinationEnv, HallucinationObservation
from openenv.core import create_web_interface_app

app = create_web_interface_app(
    HallucinationEnv,
    HallucinationAction,
    HallucinationObservation,
    env_name="Hallucination Detection Environment",
)