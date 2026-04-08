from models.observation import PipelineStage
from models.reward import Reward, RewardReason
from typing import List


CORRECT_FIX_REWARD = 0.4
PARTIAL_FIX_REWARD = 0.1
WRONG_STAGE_PENALTY = -0.2
ALREADY_FIXED_PENALTY = -0.1
TASK_COMPLETE_BONUS = 0.2
INVALID_ACTION_PENALTY = -0.3


def compute_reward(
    action_stage: PipelineStage,
    action_fix: str,
    expected_stage: PipelineStage,
    correct_keywords: List[str],
    partial_keywords: List[str],
    stages_fixed: List[PipelineStage],
    all_stages_done: bool,
    cumulative: float,
) -> Reward:
    fix_lower = action_fix.lower()

    # Already fixed this stage
    if action_stage in stages_fixed:
        return Reward(
            value=ALREADY_FIXED_PENALTY,
            reason=RewardReason.ALREADY_FIXED,
            detail=f"Stage '{action_stage}' was already fixed.",
            cumulative=round(cumulative + ALREADY_FIXED_PENALTY, 2),
        )

    # Wrong stage targeted
    if action_stage != expected_stage:
        return Reward(
            value=WRONG_STAGE_PENALTY,
            reason=RewardReason.WRONG_STAGE,
            detail=f"Expected stage '{expected_stage}', but agent targeted '{action_stage}'.",
            cumulative=round(cumulative + WRONG_STAGE_PENALTY, 2),
        )

    # Correct stage — check fix quality
    is_correct = any(kw.lower() in fix_lower for kw in correct_keywords)
    is_partial = any(kw.lower() in fix_lower for kw in partial_keywords)

    if is_correct:
        value = CORRECT_FIX_REWARD
        if all_stages_done:
            value += TASK_COMPLETE_BONUS
        reason = RewardReason.CORRECT_FIX if not all_stages_done else RewardReason.TASK_COMPLETE
        return Reward(
            value=value,
            reason=reason,
            detail="Correct fix applied." + (" All stages complete — bonus awarded!" if all_stages_done else ""),
            cumulative=round(cumulative + value, 2),
        )
    elif is_partial:
        return Reward(
            value=PARTIAL_FIX_REWARD,
            reason=RewardReason.PARTIAL_FIX,
            detail="Partially correct fix. Right direction but missing key detail.",
            cumulative=round(cumulative + PARTIAL_FIX_REWARD, 2),
        )
    else:
        return Reward(
            value=WRONG_STAGE_PENALTY,
            reason=RewardReason.WRONG_STAGE,
            detail="Fix did not match any expected correct or partial keywords.",
            cumulative=round(cumulative + WRONG_STAGE_PENALTY, 2),
        )