"""Scoring for submitted inference policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_opt.budget import Budget
from inference_opt.evaluator import evaluate_policy


def policy_score(config: dict[str, Any], work_dir: str):
    """Build a scorer that independently evaluates a submitted artifact."""

    def score_fn(result: str) -> float:
        try:
            path = Path(result.strip())
            if not path.is_absolute():
                path = Path(work_dir) / path
            budget = Budget(max_student_calls=config["final_max_student_calls"])
            evaluation = evaluate_policy(
                policy_path=path,
                data_dir=config["data_dir"],
                benchmark=config["benchmark"],
                models=list(config["models"]),
                split="test",
                budget=budget,
            )
            baselines = config.get("baselines", {})
            improvements = [
                evaluation["models"][model]["score"] - float(baselines.get(model, 0.0))
                for model in config["models"]
            ]
            value = min(improvements) if config.get("joint", False) else sum(improvements) / len(improvements)
            return float(max(0.0, min(1.0, value + config.get("offset", 0.0))))
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            return 0.0

    return score_fn
