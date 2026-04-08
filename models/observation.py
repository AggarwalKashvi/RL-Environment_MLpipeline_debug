from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from openenv.core.env_server import Observation as OpenEnvObservation

class PipelineStage(str, Enum):
    PREPROCESSING = "preprocessing"
    MODEL_CONFIG = "model_config"
    TRAINING_LOOP = "training_loop"


class BugSymptom(BaseModel):
    stage: PipelineStage
    symptom: str
    code_snippet: str


class Observation(BaseModel):
    task_id: str
    step: int
    current_stage: PipelineStage
    bug_symptom: BugSymptom
    pipeline_context: str
    stages_fixed: List[PipelineStage] = Field(default_factory=list)
    stages_remaining: List[PipelineStage]
    total_stages: int
    message: Optional[str] = None
    reward: float = 0.0
    done: bool = False

    # This config block is the secret sauce for Pydantic v2
    model_config = {
        "extra": "allow",
        "use_enum_values": True,
        "populate_by_name": True
    }