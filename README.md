---
title: ML Pipeline Debugger
emoji: 🐛
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# ML Pipeline Debugger Environment

## Motivation & Real-World Utility
Machine learning pipelines frequently break in production due to silent data leakage, improper tensor shapes, or misconfigured loss functions. This environment simulates the real-world task of an MLOps engineer or AI agent debugging a broken machine learning pipeline. Instead of a toy game, the agent must parse Python code snippets, understand the pipeline context, and inject the correct mathematical or API fixes to unblock the pipeline.

## Observation Space
At each step, the agent receives a rich state dictionary detailing the pipeline's failure:
* **`task_id`**: The current debugging scenario.
* **`current_stage`**: The stage currently throwing an error (Preprocessing, Model Config, or Training Loop).
* **`bug_symptom`**: A description of the symptom (e.g., "Training loss is NaN") and the raw `code_snippet` causing the issue.
* **`pipeline_context`**: High-level context (e.g., "A multi-class CNN pipeline").
* **`stages_fixed` & `stages_remaining`**: Progress tracking through the pipeline.

## Action Space
The agent acts by submitting a JSON payload representing a patch to the codebase:
* **`stage`**: The pipeline stage the agent intends to fix.
* **`fix`**: The specific code change or logic required (e.g., "optimizer.zero_grad()").
* **`reason`**: The agent's justification for the fix.
* **`confidence`**: A float (0.0 - 1.0) representing the agent's certainty.

## Tasks & Difficulty Progression
1. **Task 1 (Easy) - The Silent Leak:** * **Bug:** Data leakage caused by calling `fit_transform` on a test set.
   * **Goal:** Agent must identify the leakage and replace it with `transform()`.
2. **Task 2 (Medium) - The Exploding Loss:** * **Bugs:** Missing image normalization (uint8 scale) and applying `BCELoss` to a multi-class problem.
   * **Goal:** Agent must normalize the tensors and switch to `CrossEntropyLoss`.
3. **Task 3 (Hard) - The Cascading Failure:** * **Bugs:** Bruteforce `dropna()` destroying the dataset, a massively oversized learning rate, and a missing gradient zeroing step in the PyTorch training loop.
   * **Goal:** Agent must impute NaNs, lower the learning rate, and add `optimizer.zero_grad()`.

## Reward Shaping
The environment uses a deterministic grader that rewards partial progress to create a smooth learning trajectory:
* **+0.4 (Correct Fix):** The agent provided the exact correct syntax.
* **+0.1 (Partial Fix):** The agent identified the right concept (e.g., "impute") but missed the exact implementation.
* **-0.2 (Wrong Stage):** The agent tried to patch a stage that isn't currently broken.
* **+0.2 (Bonus):** Awarded for completing the entire pipeline flawlessly.

## Setup & Local Usage
1. Build the container: `docker build -t ml-debugger .`
2. Run the environment: `docker run -p 8000:8000 ml-debugger`
3. Test an agent: `python inference.py` (Ensure `ENV_BASE_URL` is set to the local or Hugging Face Space URL).