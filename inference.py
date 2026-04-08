"""
inference.py — Agent runner for MLPipelineDebugEnv
Uses OpenAI-compatible client pointed at any LLM endpoint.

Required env vars:
    API_BASE_URL   — base URL of the LLM API (e.g. https://api.openai.com/v1)
    MODEL_NAME     — model to use (e.g. gpt-4o)
    HF_TOKEN       — Hugging Face token (used as API key if targeting HF endpoint)

Optional env vars:
    TASK_ID        — which task to run: task_1 | task_2 | task_3 (default: task_1)
    ENV_BASE_URL   — base URL of the running env API (default: http://localhost:8000)
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ------------------------------------------------------------------
# Config from environment
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TASK_ID = os.environ.get("TASK_ID", "task_1")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = 20

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)

SYSTEM_PROMPT = """You are an expert ML engineer debugging a broken ML pipeline.
At each step you will receive:
- A description of the pipeline
- The current stage with a buggy code snippet and symptom

You must respond ONLY with a valid JSON object with these exact keys:
{
  "stage": "<preprocessing|model_config|training_loop>",
  "fix": "<specific fix to apply>",
  "reason": "<why this fixes the bug>",
  "confidence": <float 0.0 to 1.0>
}

No extra text. No markdown. Only the JSON object."""


def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload or {})
    else:
        resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def obs_to_prompt(obs: dict) -> str:
    bug = obs.get("bug_symptom", {})
    return (
        f"Pipeline context: {obs.get('pipeline_context', '')}\n\n"
        f"Current stage: {obs.get('current_stage', '')}\n"
        f"Symptom: {bug.get('symptom', '')}\n\n"
        f"Buggy code:\n```python\n{bug.get('code_snippet', '')}\n```\n\n"
        f"Stages already fixed: {obs.get('stages_fixed', [])}\n"
        f"Stages remaining: {obs.get('stages_remaining', [])}\n\n"
        f"What is the fix? Respond with JSON only."
    )


def parse_action(response_text: str) -> dict:
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)


def main():
    env_name = MLPipelineDebugEnv_NAME = "MLPipelineDebugEnv"

    # [START]
    print(f"[START] task={TASK_ID} env={env_name} model={MODEL_NAME}")

    # Reset environment
    obs = call_env("reset", {"task_id": TASK_ID})

    step_num = 0
    reward_log = []
    done = False
    final_score = 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done and step_num < MAX_STEPS:
        user_msg = obs_to_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            raw = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": raw})
            action = parse_action(raw)
        except Exception as e:
            print(
                f"[STEP] step={step_num} action=null reward=-0.30 done=false error={str(e)}"
            )
            reward_log.append(-0.30)
            step_num += 1
            continue

        # Step environment
        try:
            result = call_env("step", action)
            obs = result["observation"]
            reward_val = result["reward"]["value"]
            done = result["done"]
            info = result.get("info", {})
            final_score = info.get("score") or 0.0

            reward_log.append(reward_val)

            # [STEP]
            print(
                f"[STEP] step={step_num} "
                f"action={json.dumps(action.get('fix', ''))[:60]} "
                f"reward={reward_val:.2f} "
                f"done={'true' if done else 'false'} "
                f"error=null"
            )
        except Exception as e:
            print(
                f"[STEP] step={step_num} action=null reward=-0.30 done=false error={str(e)}"
            )
            reward_log.append(-0.30)

        step_num += 1

    # Get final score if not already set
    if not final_score:
        try:
            state = call_env("state", method="GET")
            final_score = 0.0  # will be computed by grader on submission
        except Exception:
            pass

    rewards_str = ",".join(f"{r:.2f}" for r in reward_log)
    success = done and final_score >= 0.8

    # [END]
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"score={final_score:.4f} "
        f"rewards={rewards_str}"
    )


if __name__ == "__main__":
    main()