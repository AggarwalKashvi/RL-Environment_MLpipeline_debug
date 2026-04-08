from pydantic import Field, field_validator, ConfigDict
from typing import Optional
from models.observation import PipelineStage
from openenv.core.env_server import Action as OpenEnvAction

class Action(OpenEnvAction):
    stage: PipelineStage
    fix: str
    reason: str
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(
        extra="ignore",         # If AI adds extra keys, ignore them instead of crashing
        use_enum_values=True
    )

    @field_validator("stage", mode="before")
    @classmethod
    def lowercase_stage(cls, v):
        # AI often outputs "Preprocessing" (Title Case). 
        # This forces it to lowercase to match your Enum.
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("fix", "reason")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()