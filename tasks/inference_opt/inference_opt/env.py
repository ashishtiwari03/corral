"""Corral environment factory for inference-time LLM optimization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from inference_opt.score import policy_score
from inference_opt.tools import create_tools

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", ".corral/workspaces")


def load_tasks_from_json(json_path: str | Path, work_dir: str) -> dict[str, TaskDefinition]:
    """Load task metadata and bind each task to its policy scorer and tools."""
    entries: list[dict[str, Any]] = []
    path = Path(json_path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for file in files:
        value = json.loads(file.read_text())
        entries.extend(value if isinstance(value, list) else [value])

    tasks = {}
    for entry in entries:
        config = dict(entry["initial_input"])
        config["data_dir"] = os.environ.get(
            "CORRAL_INFERENCE_DATA_DIR", config.get("data_dir", "data")
        )
        tasks[entry["id"]] = TaskDefinition(
            name=entry["name"],
            description=entry["description"],
            tools=list(entry.get("tools", [])),
            scoring_fn=policy_score(config, work_dir),
            submission_format=entry["submission_format"],
            initial_input=config,
            resolve_answer=False,
        )
    return tasks


def create_environments(
    *,
    local_dir: str | Path | None = None,
    level: int = 1,
    task_type: str = "task",
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
    """Create the ten Level 1 or Level 2 inference-optimization tasks."""
    root = Path(__file__).resolve().parents[1]
    source = Path(local_dir) if local_dir is not None else root / "environments" / f"level_{level}" / ("subtasks_json" if task_type == "subtask" else "tasks_json")
    tasks = load_tasks_from_json(source, work_dir)
    # Tools are task-bound because each task has its own benchmark/model pair.
    # Building them in one shared pool would accidentally bind every task to
    # the last task's configuration.
    return {
        task_id: Environment(
            task_id=task_id,
            task=task,
            base_work_dir=work_dir,
            component_id=f"inference_opt_level_{level}",
            toolset=Toolset(
                pool=create_tools(task.initial_input, work_dir),
                workspace_factory=None,
            ),
        )
        for task_id, task in tasks.items()
    }
