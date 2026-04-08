from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from env.environment import MLPipelineDebugEnv
from models.action import Action
from models.observation import Observation
from models.reward import Reward

app = FastAPI(
    title="MLPipelineDebugEnv",
    description="OpenEnv-compatible ML pipeline debugging environment API",
    version="1.0.0",
)

# Single global environment instance (stateful)
env = MLPipelineDebugEnv()


# ------------------------------------------------------------------
# Request/Response schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"


class StepRequest(BaseModel):
    stage: str
    fix: str
    reason: str
    confidence: Optional[float] = 1.0


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.post("/reset")
def reset(request: ResetRequest):
    try:
        obs = env.reset(task_id=request.task_id)
        return {
            "observation": obs.dict(),
            "done": False,
            "info": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    try:
        from models.observation import PipelineStage
        action = Action(
            stage=PipelineStage(request.stage),
            fix=request.fix,
            reason=request.reason,
            confidence=request.confidence,
        )
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.dict(),
            reward=reward.value,
            done=done,
            info={
                "reward_reason": info["reward_reason"],
                "score": info["score"],
                "steps": env._step_count,
            },
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok", "env": MLPipelineDebugEnv.ENV_NAME}