import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional

from openenv.core.env_server import create_fastapi_app
from env.environment import MLPipelineDebugEnv
from models.action import Action
from models.observation import Observation

# ---------------------------------------------------------------------------
# Persistent environment singleton
#
# OpenEnv's default HTTP server creates a fresh env instance per request,
# so state (history, bug progress, reward log) is destroyed between calls.
# We keep ONE env instance at module level and override /reset, /step, /state
# with stateful endpoints that use it directly.
# ---------------------------------------------------------------------------
_env = MLPipelineDebugEnv()

# Base app — gives us /schema, /health, /metadata, /ws, /docs
app = create_fastapi_app(MLPipelineDebugEnv, Action, Observation)


# ---------------------------------------------------------------------------
# Stateful /reset
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "task_1"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


@app.post("/reset", tags=["Environment Control"])
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> Dict[str, Any]:
    obs = _env.reset(task_id=request.task_id)
    return obs.model_dump()


# ---------------------------------------------------------------------------
# Stateful /step
# ---------------------------------------------------------------------------
class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/step", tags=["Environment Control"])
def step(request: StepRequest) -> Dict[str, Any]:
    action = Action(**request.action)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": {},
    }


# ---------------------------------------------------------------------------
# /score — graded final score (call after done=True)
# ---------------------------------------------------------------------------
@app.get("/score", tags=["State Management"])
def get_score() -> Dict[str, Any]:
    return {"score": _env.final_score()}


# ---------------------------------------------------------------------------
# /state — current env state from singleton
# ---------------------------------------------------------------------------
@app.get("/state", tags=["State Management"])
def get_state() -> Dict[str, Any]:
    return _env.state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Starting OpenEnv server (stateful singleton mode)...")
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()