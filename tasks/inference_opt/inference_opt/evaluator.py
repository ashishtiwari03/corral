"""Generic JSONL evaluator for frozen-model inference policies."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_opt.client import PolicyModelClient, VLLMClient
from inference_opt.policy import load_policy

if TYPE_CHECKING:
    from inference_opt.budget import Budget


def load_questions(data_dir: str | Path, benchmark: str, split: str) -> list[dict[str, Any]]:
    path = Path(data_dir) / benchmark / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"question file not found: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(value: Any) -> str:
    """Normalize common free-form and multiple-choice model outputs."""
    text = str(value).strip().lower()
    match = re.search(r"(?:final answer|answer)\s*[:=]\s*([a-z])\b", text)
    if match:
        return match.group(1)
    return re.sub(r"\s+", " ", text).strip(" .`\"'")


def answer_is_correct(prediction: Any, record: dict[str, Any]) -> bool:
    expected = normalize_answer(record["answer"])
    actual = normalize_answer(prediction)
    if actual == expected:
        return True
    choices = record.get("choices") or []
    if choices:
        for index, choice in enumerate(choices):
            if actual in {normalize_answer(choice), chr(ord("a") + index)}:
                return normalize_answer(choice) == expected or chr(ord("a") + index) == expected
    return False


def endpoint_for(model: str) -> str:
    key = "CORRAL_VLLM_URL_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
    return os.environ.get(key, os.environ.get("CORRAL_VLLM_URL", "http://127.0.0.1:8000"))


def evaluate_policy(
    *,
    policy_path: str | Path,
    data_dir: str | Path,
    benchmark: str,
    models: list[str],
    split: str,
    budget: Budget,
    diagnostics: bool = False,
) -> dict[str, Any]:
    """Run a policy and return aggregate metrics plus optional diagnostics."""
    solve = load_policy(policy_path)
    records = load_questions(data_dir, benchmark, split)
    budget.reserve_experiment(0)
    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        client = PolicyModelClient(VLLMClient(endpoint_for(model), model=model, budget=budget))
        details = []
        for record in records:
            public_record = {
                key: value for key, value in record.items() if key != "answer"
            }
            answer = solve(
                record["question"],
                client,
                {"benchmark": benchmark, **public_record},
            )
            details.append({"id": record.get("id"), "correct": answer_is_correct(answer, record), "answer": str(answer)})
        score = sum(item["correct"] for item in details) / max(len(details), 1)
        by_model[model] = {"score": score, "num_questions": len(details)}
        if diagnostics:
            by_model[model]["questions"] = details
    result = {"benchmark": benchmark, "split": split, "models": by_model, "student_calls": budget.student_calls}
    result["score"] = sum(item["score"] for item in by_model.values()) / max(len(by_model), 1)
    return result
