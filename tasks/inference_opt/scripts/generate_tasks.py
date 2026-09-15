"""Generate the level 1 and level 2 task definitions.

Kept as a script rather than hand-edited JSON because the files are one object per
line across 24 tasks, and because ``measure_baselines.py --write-tasks`` has to
rewrite the ``baselines`` fields in place afterwards.

Usage::

    uv run python scripts/generate_tasks.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS: dict[str, str] = {
    "gsm8k": "GSM8K grade-school maths word problems",
    "mmlu_pro": "MMLU-Pro multiple-choice questions across 14 subjects",
    "gpqa_diamond": "GPQA-Diamond graduate-level science questions",
    "bbh": "BIG-Bench Hard reasoning tasks",
    "chembench": "ChemBench chemistry questions",
    "arc_challenge": "ARC-Challenge science questions",
}

#: A deterministic namespace, so regenerating does not churn every uuid.
NAMESPACE = uuid.UUID("6f1d5a52-0f6e-4a1d-9a27-6b6c9f0e1a10")

LEVEL_1_MODELS = [["model_a"], ["model_b"]]
LEVEL_2_MODELS = [["model_a", "model_b"], ["model_c", "model_d"]]

TOOLS = [
    "get_baseline",
    "reveal_train_questions",
    "query_student",
    "dry_run_policy",
    "evaluate_candidate",
    "inspect_failures",
    "compare_runs",
    "get_budget",
    "submit_policy",
]

N_TRAIN = 30
N_TEST = 30


def _task(benchmark: str, models: list[str], level: int) -> dict:
    joint = len(models) > 1
    suffix = "".join(model.rsplit("_", 1)[-1] for model in models)
    task_id = f"{benchmark}_{suffix}"
    # Test-time allowance is per model, so a joint policy cannot starve one model
    # to buy compute for the other and game the level-2 minimum.
    calls_per_question = 8
    final_calls = N_TEST * calls_per_question

    if joint:
        name = f"Joint {benchmark} improvement on {' and '.join(models)}"
        description = (
            f"Build one shared inference-time policy that improves the frozen "
            f"students {' and '.join(models)} on {BENCHMARKS[benchmark]}. "
            f"You are scored on the smaller of the two improvements, so the policy "
            f"must help both models, not trade one off against the other."
        )
    else:
        name = f"Improve {benchmark} on {models[0]}"
        description = (
            f"Build and submit an inference-time policy that improves the frozen "
            f"student {models[0]} on {BENCHMARKS[benchmark]}. Do not modify model "
            f"weights; change only what happens around the model at test time."
        )

    return {
        "id": task_id,
        "uuid": str(uuid.uuid5(NAMESPACE, f"inference-opt-l{level}-{task_id}")),
        "name": name,
        "description": description,
        "tools": TOOLS,
        "submission_format": (
            "The literal string 'submission.json' - stage it with submit_policy first."
        ),
        "initial_input": {
            "benchmark": benchmark,
            "models": models,
            "joint": joint,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            # Filled in by scripts/measure_baselines.py --write-tasks.
            "baselines": dict.fromkeys(models, 0.0),
            "baselines_train": dict.fromkeys(models, 0.0),
            "baseline_items": {},
            "model_specs": {},
            "base_urls": {},
            "max_calls_per_question": calls_per_question,
            "final_max_calls_per_question": 12,
            "final_max_student_calls": final_calls,
            "final_setup_calls": 0,
            "setup_calls": 20,
            "scale": 1.0,
            "budget": {
                "max_experiments": 20,
                "max_debug_runs": 10,
                "max_student_calls": 20 * N_TRAIN * calls_per_question // 4,
                "max_reveals": 4,
                "max_probe_calls": 40,
                "reveal_batch": 5,
            },
        },
    }


def main() -> None:
    for level, rosters in ((1, LEVEL_1_MODELS), (2, LEVEL_2_MODELS)):
        tasks = [
            _task(benchmark, models, level)
            for benchmark in BENCHMARKS
            for models in rosters
        ]
        target = ROOT / "environments" / f"level_{level}" / "tasks_json"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "tasks.json"
        # One object per line: readable in a terminal, and diffs stay per-task.
        body = "[\n" + ",\n".join(
            "  " + json.dumps(task, sort_keys=True) for task in tasks
        ) + "\n]\n"
        path.write_text(body, encoding="utf-8")
        print(f"level {level}: wrote {len(tasks)} tasks to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
