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
import json
import requests
from openai import OpenAI

# ------------------------------------------------------------------
# Config from environment
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
TASK_ID      = os.environ.get("TASK_ID", "task_1")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS    = 20
ENV_NAME     = "MLPipelineDebugEnv"

llm = OpenAI(
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

CRITICAL RULES:
- The "stage" value MUST exactly match the current stage shown to you.
- Fix ONE stage at a time in the order given. Do NOT skip ahead.
- No extra text. No markdown. Only the raw JSON object."""


def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload or {})
    else:
        resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def obs_to_prompt(obs: dict) -> str:
    current_stage = obs.get("current_stage", "")
    return (
        f"Pipeline context: {obs.get('pipeline_context', '')}\n\n"
        f">>> CURRENT STAGE TO FIX: {current_stage} <<<\n"
        f"You MUST set stage to exactly: \"{current_stage}\"\n\n"
        f"Symptom: {obs.get('symptom', '')}\n\n"
        f"Buggy code:\n```python\n{obs.get('code_snippet', '')}\n```\n\n"
        f"Stages already fixed: {obs.get('stages_fixed', [])}\n"
        f"Stages remaining: {obs.get('stages_remaining', [])}\n\n"
        f"Respond with JSON only. stage must be \"{current_stage}\"."
    )

def parse_action(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)


def fetch_final_score() -> float:
    """Call /score endpoint if available, else fall back to /state."""
    # Try dedicated /score endpoint first
    try:
        result = call_env("score", method="GET")
        return float(result.get("score", 0.0))
    except Exception:
        pass

    # Fall back: read cumulative_reward from /state as a proxy
    try:
        state = call_env("state", method="GET")
        # Return grader score if present, else 0
        return float(state.get("final_score", 0.0))
    except Exception:
        return 0.0


def main():
    print(f"[START] task={TASK_ID} env={ENV_NAME} model={MODEL_NAME}")

    obs = call_env("reset", {"task_id": TASK_ID})

    step_num    = 0
    reward_log  = []
    done        = False
    final_score = 0.0
    messages    = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done and step_num < MAX_STEPS:
        messages.append({"role": "user", "content": obs_to_prompt(obs)})

        # ── Call LLM ──────────────────────────────────────────────
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            raw = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": raw})
            action = parse_action(raw)
        except Exception as e:
            print(f"[STEP] step={step_num} action=null reward=-0.30 done=false error={e}")
            reward_log.append(-0.30)
            step_num += 1
            continue

        # ── Step environment ───────────────────────────────────────
        try:
            result     = call_env("step", action)
            obs        = result["observation"]
            reward_val = result["reward"]["value"]
            done       = result["done"]

            # Score is returned by the env on every step (not just final)
            score_from_step = result.get("info", {}).get("score")
            if score_from_step is not None:
                final_score = float(score_from_step)

            reward_log.append(reward_val)

            # Reset message history on correct fix so model doesn't get stuck
            if reward_val > 0.3:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            print(
                f"[STEP] step={step_num} "
                f"action={json.dumps(action.get('fix', ''))[:60]} "
                f"reward={reward_val:.2f} "
                f"done={'true' if done else 'false'} "
                f"error=null"
            )
        except Exception as e:
            print(f"[STEP] step={step_num} action=null reward=-0.30 done=false error={e}")
            reward_log.append(-0.30)

        step_num += 1

    # ── Fetch real grader score if episode finished ────────────────
    if done and final_score == 0.0:
        final_score = fetch_final_score()

    rewards_str = ",".join(f"{r:.2f}" for r in reward_log)
    success = done and final_score >= 0.8

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"score={final_score:.4f} "
        f"rewards={rewards_str}"
    )


if __name__ == "__main__":
    main()