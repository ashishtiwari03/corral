"""Corral tools exposed to the teacher agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corral.core.tool import Tool, tool
from inference_opt.budget import Budget
from inference_opt.client import PolicyModelClient, VLLMClient
from inference_opt.evaluator import (
    answer_is_correct,
    endpoint_for,
    evaluate_policy,
    load_questions,
)
from inference_opt.policy import load_policy, validate_policy


def create_tools(config: dict[str, Any], work_dir: str) -> dict[str, Tool]:
    """Create tools bound to one task's benchmark, models, and workspace."""
    budget = Budget(**config.get("budget", {}))
    data_dir = config["data_dir"]

    @tool
    def get_baseline() -> str:
        """Return the frozen-model baseline scores configured for this task."""
        return json.dumps(config.get("baselines", {}), sort_keys=True)

    @tool
    def evaluate_candidate(policy_path: str, split: str = "development") -> str:
        """Evaluate a candidate policy on the permitted benchmark split."""
        result = evaluate_policy(
            policy_path=Path(work_dir) / policy_path,
            data_dir=data_dir,
            benchmark=config["benchmark"],
            models=list(config["models"]),
            split=split,
            budget=budget,
            diagnostics=split == "development",
        )
        return json.dumps(result, sort_keys=True)

    @tool
    def run_single_question(policy_path: str, question_id: str, model: str) -> str:
        """Run one development question for targeted debugging."""
        records = load_questions(data_dir, config["benchmark"], "development")
        matches = [record for record in records if str(record.get("id")) == question_id]
        if not matches:
            raise ValueError(f"unknown development question: {question_id}")
        budget.reserve_debug(0)
        record = matches[0]
        public_record = {
            key: value for key, value in record.items() if key != "answer"
        }
        answer = load_policy(Path(work_dir) / policy_path)(
            record["question"],
            PolicyModelClient(VLLMClient(endpoint_for(model), model=model, budget=budget)),
            {"benchmark": config["benchmark"], **public_record},
        )
        return json.dumps({"id": question_id, "answer": str(answer), "correct": answer_is_correct(answer, record), "budget": budget.snapshot()}, sort_keys=True)

    @tool
    def compare_runs(run_summaries: str) -> str:
        """Compare JSON evaluation summaries supplied from earlier experiments."""
        summaries = json.loads(run_summaries)
        if not isinstance(summaries, list):
            raise ValueError("run_summaries must be a JSON list")
        return json.dumps({"runs": len(summaries), "best_score": max((item.get("score", 0.0) for item in summaries), default=0.0), "budget": budget.snapshot()}, sort_keys=True)

    @tool
    def get_budget() -> str:
        """Return remaining teacher experiment and student inference budget."""
        return json.dumps(budget.snapshot(), sort_keys=True)

    @tool
    def submit_policy(policy_path: str) -> str:
        """Validate and submit the final policy artifact for hidden evaluation."""
        resolved = validate_policy(Path(work_dir) / policy_path)
        return json.dumps({"submitted": True, "policy_path": str(resolved), "budget": budget.snapshot()}, sort_keys=True)

    return {item.name: item for item in (get_baseline, evaluate_candidate, run_single_question, compare_runs, get_budget, submit_policy)}
