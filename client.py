from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models.action import Action
from models.observation import Observation

class MLPipelineClient(EnvClient[Action, Observation, Dict[str, Any]]):
    
    def _step_payload(self, action: Action) -> dict:
        # Translates the Python Action object into a JSON payload
        return {
            "stage": action.stage.value if hasattr(action.stage, 'value') else action.stage,
            "fix": action.fix,
            "reason": action.reason,
            "confidence": action.confidence
        }

    def _parse_result(self, payload: dict) -> StepResult:
        # Translates the JSON response back into your Python Observation object
        obs_data = payload.get("observation", {})
        
        # OpenEnv expects a StepResult object combining observation, reward, and done
        return StepResult(
            observation=Observation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> Dict[str, Any]:
        # Returns the raw state dictionary
        return payload