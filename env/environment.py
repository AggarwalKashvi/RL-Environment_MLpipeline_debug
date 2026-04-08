from typing import Optional, Dict, Any, List
from openenv.core.env_server import Environment   # <-- ADDED IMPORT

from models.observation import Observation, BugSymptom, PipelineStage
from models.action import Action
from models.reward import Reward
from env.tasks import TASKS
from env.rewards import compute_reward
from env.graders import grade_task

# 1. ADDED INHERITANCE
class MLPipelineDebugEnv(Environment):
    ENV_NAME = "MLPipelineDebugEnv"

    def __init__(self):
        self._task_id: Optional[str] = None
        self._bugs: List[Dict] = []
        self._current_bug_index: int = 0
        self._step_count: int = 0
        self._stages_fixed: List[PipelineStage] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []
        self._reward_log: List[float] = []

    # 2. UPDATED SIGNATURE (OpenEnv passes seed and episode_id automatically)
    def reset(self, seed=None, episode_id=None, task_id: str = "task_1", **kwargs) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")

        task = TASKS[task_id]
        self._task_id = task_id
        self._bugs = task["bugs"]
        self._current_bug_index = 0
        self._step_count = 0
        self._stages_fixed = []
        self._cumulative_reward = 0.0
        self._done = False
        self._history = []
        self._reward_log = []

        obs = self._make_observation()
        # Ensure base OpenEnv fields are populated on reset
        obs.done = False
        obs.reward = 0.0 
        return obs

    # 3. UPDATED SIGNATURE (Returns ONLY Observation)
    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        if not self._bugs:
            # Re-initialize the task if memory was wiped
            self.reset(task_id=self._task_id or "task_1")
        
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_bug = self._bugs[self._current_bug_index]
        expected_stage = current_bug["stage"]
        all_stages_will_be_done = (
            action.stage == expected_stage and
            len(self._stages_fixed) + 1 == len(self._bugs)
        )

        reward = compute_reward(
            action_stage=action.stage,
            action_fix=action.fix,
            expected_stage=expected_stage,
            correct_keywords=current_bug["correct_keywords"],
            partial_keywords=current_bug["partial_keywords"],
            stages_fixed=self._stages_fixed,
            all_stages_done=all_stages_will_be_done,
            cumulative=self._cumulative_reward,
        )

        self._cumulative_reward = reward.cumulative
        self._step_count += 1
        self._reward_log.append(reward.value)

        self._history.append({
            "stage": action.stage,
            "fix": action.fix,
            "reason": action.reason,
            "reward": reward.value,
            "reward_reason": reward.reason,
        })

        from models.reward import RewardReason
        if reward.reason in (RewardReason.CORRECT_FIX, RewardReason.TASK_COMPLETE):
            self._stages_fixed.append(action.stage)
            self._current_bug_index += 1

        if self._current_bug_index >= len(self._bugs):
            self._done = True

        # Generate the observation
        obs = self._make_observation(message=reward.detail)
        
        # 4. ATTACH REWARD AND DONE DIRECTLY TO OBSERVATION
        obs.reward = reward.value
        obs.done = self._done
        
        return obs

    # 5. ADDED @property DECORATOR (Required by OpenEnv spec)
    @property
    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "current_bug_index": self._current_bug_index,
            "stages_fixed": [s.value for s in self._stages_fixed],
            "stages_remaining": [
                b["stage"].value for b in self._bugs[self._current_bug_index:]
            ],
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "reward_log": self._reward_log,
        }

    # ------------------------------------------------------------------
    # final_score()
    # ------------------------------------------------------------------
    def final_score(self) -> float:
        return grade_task(self._task_id, self._history, self._bugs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_observation(self, message: Optional[str] = None) -> Observation:
        if self._done:
            # Return terminal observation
            last_bug = self._bugs[-1]
            return Observation(
                task_id=self._task_id,
                step=self._step_count,
                current_stage=last_bug["stage"],
                bug_symptom=BugSymptom(
                    stage=last_bug["stage"],
                    symptom="All bugs fixed.",
                    code_snippet="# Pipeline is clean.",
                ),
                pipeline_context=TASKS[self._task_id]["pipeline_context"],
                stages_fixed=self._stages_fixed,
                stages_remaining=[],
                total_stages=len(self._bugs),
                message="Task complete! All pipeline bugs resolved.",
            )

        current_bug = self._bugs[self._current_bug_index]
        remaining_stages = [b["stage"] for b in self._bugs[self._current_bug_index:]]

        return Observation(
            task_id=self._task_id,
            step=self._step_count,
            current_stage=current_bug["stage"],
            bug_symptom=BugSymptom(
                stage=current_bug["stage"],
                symptom=current_bug["symptom"],
                code_snippet=current_bug["code_snippet"],
            ),
            pipeline_context=TASKS[self._task_id]["pipeline_context"],
            stages_fixed=self._stages_fixed,
            stages_remaining=remaining_stages,
            total_stages=len(self._bugs),
            message=message,
        )