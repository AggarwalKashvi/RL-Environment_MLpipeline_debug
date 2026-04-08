from pydantic import BaseModel, Field, ConfigDict
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

class Observation(OpenEnvObservation):
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

    # Using ConfigDict is the official Pydantic V2 way
    model_config = ConfigDict(
        extra="allow",           # Allows the server to attach metadata
        use_enum_values=True,    # Converts Enums to strings for the AI
        populate_by_name=True
    )