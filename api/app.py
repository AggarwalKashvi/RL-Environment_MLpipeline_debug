from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from env.environment import MLPipelineDebugEnv
from models.action import Action
from models.observation import PipelineStage

app = FastAPI(
    title="MLPipelineDebugEnv",
    description="OpenEnv-compatible ML pipeline debugging environment API",
    version="1.0.0",
)

env = MLPipelineDebugEnv()


# ── Request schemas ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"


class StepRequest(BaseModel):
    stage: str
    fix: str
    reason: str
    confidence: Optional[float] = 1.0


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest):
    try:
        obs = env.reset(task_id=request.task_id)
        return obs.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    try:
        action = Action(
            stage=PipelineStage(request.stage),
            fix=request.fix,
            reason=request.reason,
            confidence=request.confidence,
        )
        obs, reward, done, info = env.step(action)

        # Always compute and return the grader score, not just when done
        current_score = env.final_score()

        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": {
                "reward_reason": info["reward_reason"],
                "score": current_score,   # ← real grader score every step
                "steps": env._step_count,
            },
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
def state():
    s = env.state()
    s["final_score"] = env.final_score()   # ← include grader score in state too
    return s


@app.get("/score")
def score():
    """Dedicated endpoint for fetching the current grader score."""
    return {
        "score": env.final_score(),
        "steps": env._step_count,
        "done": env._done,
        "cumulative_reward": env._cumulative_reward,
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": MLPipelineDebugEnv.ENV_NAME}