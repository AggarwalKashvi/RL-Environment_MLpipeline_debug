from pydantic import BaseModel, Field, validator
from typing import Optional
from models.observation import PipelineStage
from openenv.core.env_server import Action as OpenEnvAction


class Action(BaseModel):
    stage: PipelineStage
    fix: str
    reason: str
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

    @validator("fix")
    def fix_not_empty(cls, v):
        if not v.strip():
            raise ValueError("fix cannot be empty")
        return v.strip()

    @validator("reason")
    def reason_not_empty(cls, v):
        if not v.strip():
            raise ValueError("reason cannot be empty")
        return v.strip()