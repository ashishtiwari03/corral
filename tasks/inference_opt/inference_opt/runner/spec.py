"""The controller → eval-host interface.

One JSON file in, three artifacts out. Deliberately boring: the two processes share
no Python objects, because the eval host imports policy code and the controller
holds the labels, and keeping that boundary crossable only by serialisable data is
what makes the separation real.

Note what is *absent* from :class:`RunSpec`: any path to the gold answers. The host
is handed a temporary JSONL of public records that the controller wrote, and nothing
else. A policy running in that process has nothing to read even if it tries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunSpec", "RunSummary"]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything the eval host needs to run one policy over one set of questions."""

    run_id: str
    policy_dir: str
    #: JSONL of public records. Written by the controller; contains no targets.
    questions_path: str
    out_dir: str

    #: e.g. ``"vllm/Qwen/Qwen2.5-7B-Instruct"`` or ``"mockllm/model"``.
    model_spec: str = "mockllm/model"
    #: Must end in ``/v1``. When this is unset for a ``vllm/`` model, inspect's
    #: provider starts a *new* local vLLM server rather than using ours.
    base_url: str | None = None
    api_key: str | None = None

    total_calls: int = 250
    max_calls_per_question: int = 8
    max_tokens_per_call: int = 2048
    setup_calls: int = 0

    #: Labeled train examples for ``Policy.setup``. Only ever set for train runs.
    revealed_path: str | None = None
    #: Overrides the policy manifest, for the dry-run path.
    execution_override: str | None = None
    max_connections: int = 8
    time_limit_s: int = 1800
    epochs: int = 1
    seed: int = 0
    benchmark: str = ""
    split: str = "train"

    @property
    def log_dir(self) -> str:
        return str(Path(self.out_dir) / "log")

    @property
    def predictions_path(self) -> str:
        return str(Path(self.out_dir) / "predictions.jsonl")

    @property
    def summary_path(self) -> str:
        return str(Path(self.out_dir) / "summary.json")

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> RunSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RunSummary:
    """What the eval host reports back. Written to ``summary.json``."""

    run_id: str
    ok: bool = False
    error: str = ""
    n_questions: int = 0
    n_answered: int = 0
    n_crashed: int = 0
    n_unparseable: int = 0
    calls_used: int = 0
    output_tokens: int = 0
    setup_calls_used: int = 0
    seconds: float = 0.0
    execution: str = "parallel"
    manifest: dict[str, Any] = field(default_factory=dict)
    forced_sequential: bool = False
    budget_exhausted_at: dict[str, Any] | None = None
    questions_after_exhaustion: int = 0
    per_component: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    first_tracebacks: list[str] = field(default_factory=list)
    setup_log: list[str] = field(default_factory=list)

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> RunSummary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        return cls(**known)
