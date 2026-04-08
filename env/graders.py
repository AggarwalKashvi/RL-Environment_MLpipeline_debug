from typing import List, Dict, Any


def _keyword_match_score(fix: str, correct_keywords: List[str], partial_keywords: List[str]) -> float:
    fix_lower = fix.lower()
    if any(kw.lower() in fix_lower for kw in correct_keywords):
        return 1.0
    if any(kw.lower() in fix_lower for kw in partial_keywords):
        return 0.5
    return 0.0


def grade_task(task_id: str, history: List[Dict[str, Any]], bugs: List[Dict]) -> float:
    """
    Deterministic grader for any task.
    history: list of dicts with keys: stage, fix
    bugs: list of bug dicts from TASKS (each has stage, correct_keywords, partial_keywords)
    Returns score in [0.0, 1.0]
    """
    if not history or not bugs:
        return 0.0

    total_bugs = len(bugs)
    total_score = 0.0

    for bug in bugs:
        stage = bug["stage"]
        correct_kw = bug["correct_keywords"]
        partial_kw = bug["partial_keywords"]

        # Find the best fix attempt for this stage in history
        best = 0.0
        for entry in history:
            if entry.get("stage") == stage:
                score = _keyword_match_score(entry.get("fix", ""), correct_kw, partial_kw)
                best = max(best, score)

        total_score += best

    raw = total_score / total_bugs

    # Efficiency bonus: if solved in minimum steps (equal to number of bugs), add 5%
    min_steps = total_bugs
    actual_steps = len(history)
    if actual_steps <= min_steps:
        raw = min(1.0, raw + 0.05)

    return round(raw, 4)


def grade_task_1(history: List[Dict[str, Any]], bugs: List[Dict]) -> float:
    return grade_task("task_1", history, bugs)


def grade_task_2(history: List[Dict[str, Any]], bugs: List[Dict]) -> float:
    return grade_task("task_2", history, bugs)


def grade_task_3(history: List[Dict[str, Any]], bugs: List[Dict]) -> float:
    return grade_task("task_3", history, bugs)