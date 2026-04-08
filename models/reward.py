from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RewardReason(str, Enum):
    CORRECT_FIX = "correct_fix"               # Right stage, right fix → +0.4
    PARTIAL_FIX = "partial_fix"               # Right stage, wrong/incomplete fix → +0.1
    WRONG_STAGE = "wrong_stage"               # Targeted wrong stage → -0.2
    ALREADY_FIXED = "already_fixed"           # Revisited a stage already fixed → -0.1
    TASK_COMPLETE = "task_complete"           # All bugs fixed → bonus +0.2
    INVALID_ACTION = "invalid_action"         # Malformed or unparseable action → -0.3


class Reward(BaseModel):
    value: float = Field(..., description="Numeric reward value for this step")
    reason: RewardReason = Field(..., description="Categorical reason for the reward")
    detail: Optional[str] = Field(None, description="Human-readable explanation of why this reward was given")
    cumulative: float = Field(..., description="Total cumulative reward so far in this episode")

    class Config:
        use_enum_values = True