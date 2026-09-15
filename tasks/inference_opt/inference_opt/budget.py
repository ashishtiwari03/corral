"""Budget accounting, in two layers.

The environment enforces budgets twice, because neither layer alone is sufficient:

* :class:`BudgetLedger` is **durable** and lives in a workspace file. It is the
  controller's source of truth for what the teacher agent has spent. It has to be a
  file: a non-``trusted`` Corral tool is cloudpickled into a fresh subprocess per
  call and only its return value comes back, so any counter held in a
  ``create_tools()`` closure is silently discarded after every call. That isolation
  is only active under Docker, so an in-memory design passes every local test and
  loses state in the real benchmark — which is exactly the bug this class exists to
  prevent.

* :class:`Budget` is the **in-run meter** inside the eval host. It enforces per-run
  and per-question limits while a policy executes. It is not durable and is not
  trusted on its own; the controller cross-checks the call count it reports against
  the model events recorded in the inspect log.

Calls are the headline currency, but output tokens are metered too: a single request
with a huge ``max_tokens`` can carry a dozen chained rollouts, so counting calls
alone is gameable.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_opt.api import BudgetExhausted

__all__ = [
    "Budget",
    "BudgetExhausted",
    "BudgetLedger",
    "StateLedger",
    "BudgetSpec",
    "QuestionAllocator",
    "RunRecord",
]

LEDGER_RELATIVE_PATH = Path("state") / "ledger.json"

#: Fraction of the call budget guaranteed to individual questions. The remainder is
#: a shared pool, so an adaptive policy can spend more on questions it finds hard
#: without being able to starve the ones that come later.
RESERVED_FRACTION = 0.7


@dataclass
class Budget:
    """In-memory meter for one run. Reserve before spending, never after.

    Retains the original scaffold's method names and its
    ``RuntimeError("... budget exhausted")`` contract; :class:`BudgetExhausted`
    derives from ``RuntimeError`` so existing callers and defensively written
    policies both still work.
    """

    max_experiments: int = 20
    max_debug_questions: int = 10
    max_student_calls: int = 250
    max_output_tokens: int = 0  # 0 means "derive from calls x max_tokens_per_call"
    experiments: int = 0
    debug_questions: int = 0
    student_calls: int = 0
    output_tokens: int = 0

    def reserve_experiment(self, calls: int) -> None:
        if self.experiments >= self.max_experiments:
            raise BudgetExhausted(
                f"experiment budget exhausted ({self.max_experiments} used)"
            )
        self._reserve_calls(calls)
        self.experiments += 1

    def reserve_student_call(self, count: int = 1) -> None:
        """Charge ``count`` actual student requests."""
        self._reserve_calls(count)

    def reserve_debug(self, calls: int = 1) -> None:
        if self.debug_questions >= self.max_debug_questions:
            raise BudgetExhausted(
                f"debug-question budget exhausted ({self.max_debug_questions} used)"
            )
        self._reserve_calls(calls)
        self.debug_questions += 1

    def record_tokens(self, tokens: int) -> None:
        """Account output tokens after a response returns."""
        self.output_tokens += max(0, tokens)
        if 0 < self.max_output_tokens < self.output_tokens:
            raise BudgetExhausted(
                f"output-token budget exhausted "
                f"({self.output_tokens}/{self.max_output_tokens})"
            )

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_student_calls - self.student_calls)

    def _reserve_calls(self, calls: int) -> None:
        if calls < 0:
            raise ValueError("cannot reserve a negative number of calls")
        if self.student_calls + calls > self.max_student_calls:
            raise BudgetExhausted(
                f"student inference budget exhausted "
                f"({self.student_calls}/{self.max_student_calls} used, "
                f"{calls} more requested)"
            )
        self.student_calls += calls

    def snapshot(self) -> dict[str, int]:
        return {
            "experiments": self.experiments,
            "debug_questions": self.debug_questions,
            "student_calls": self.student_calls,
            "output_tokens": self.output_tokens,
            "max_experiments": self.max_experiments,
            "max_debug_questions": self.max_debug_questions,
            "max_student_calls": self.max_student_calls,
        }


class QuestionAllocator:
    """Splits a run's call budget into per-question reserves plus a shared pool.

    Without this, a single greedy question consumed in nondeterministic order under
    concurrent execution can exhaust the run and leave later questions unattempted.
    """

    def __init__(self, total_calls: int, questions: int, per_question_cap: int) -> None:
        if questions < 1:
            raise ValueError("a run needs at least one question")
        self.total_calls = max(0, total_calls)
        self.questions = questions
        self.per_question_cap = max(1, per_question_cap)
        self.reserve_each = int(self.total_calls * RESERVED_FRACTION) // questions
        self.pool = self.total_calls - self.reserve_each * questions
        self._used: dict[str, int] = {}
        self._lock = threading.RLock()

    def allowance(self, question_id: str) -> int:
        """Calls this question may still spend, given its reserve and the pool."""
        with self._lock:
            used = self._used.get(question_id, 0)
            reserve_left = max(0, self.reserve_each - used)
            cap_left = max(0, self.per_question_cap - used)
            return min(cap_left, reserve_left + max(0, self.pool))

    def charge(self, question_id: str, calls: int) -> None:
        """Consume ``calls`` for a question, drawing from its reserve then the pool."""
        with self._lock:
            used = self._used.get(question_id, 0)
            if used + calls > self.per_question_cap:
                raise BudgetExhausted(
                    f"per-question cap reached for {question_id} "
                    f"({used}/{self.per_question_cap} calls)"
                )
            reserve_left = max(0, self.reserve_each - used)
            from_pool = max(0, calls - reserve_left)
            if from_pool > self.pool:
                raise BudgetExhausted(
                    f"run call budget exhausted at question {question_id} "
                    f"(shared pool empty, {calls} more requested)"
                )
            self.pool -= from_pool
            self._used[question_id] = used + calls

    @property
    def used_total(self) -> int:
        with self._lock:
            return sum(self._used.values())


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Immutable limits for one teacher task, read from the task config."""

    max_experiments: int = 20
    max_debug_runs: int = 10
    max_student_calls: int = 250
    max_reveals: int = 4
    max_probe_calls: int = 30
    reveal_batch: int = 5

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> BudgetSpec:
        if not raw:
            return cls()
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        # The scaffold spelled this `max_debug_questions`; accept both.
        if "max_debug_questions" in raw and "max_debug_runs" not in known:
            known["max_debug_runs"] = raw["max_debug_questions"]
        return cls(**known)


@dataclass
class RunRecord:
    """One evaluation the agent performed, as recorded in the ledger."""

    run_id: str
    kind: str  # "experiment" | "dry_run"
    policy_dir: str
    score: float | None = None
    baseline: float | None = None
    delta: float | None = None
    n_items: int = 0
    n_correct: int = 0
    calls_used: int = 0
    mcnemar_b: int = 0
    mcnemar_c: int = 0
    note: str = ""
    error: str = ""
    created_at: str = ""


@dataclass
class BudgetLedger:
    """Durable, file-backed spend and run history for one task workspace.

    Every mutation rewrites ``state/ledger.json`` atomically via ``os.replace``, so a
    tool process that dies mid-write cannot corrupt it.
    """

    path: Path
    spec: BudgetSpec = field(default_factory=BudgetSpec)
    experiments: int = 0
    debug_runs: int = 0
    student_calls: int = 0
    reveals: int = 0
    probe_calls: int = 0
    revealed_ids: list[str] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    best_run_id: str | None = None

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, work_dir: Path | str, spec: BudgetSpec | None = None) -> BudgetLedger:
        """Read the ledger for a workspace, creating an empty one if absent."""
        root = Path(work_dir).expanduser()
        path = root / LEDGER_RELATIVE_PATH
        if not path.is_file():
            return cls(path=path, spec=spec or BudgetSpec())
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt ledger must not silently reset the agent's spend to zero;
            # treat it as fully consumed so the failure is visible and safe.
            ledger = cls(path=path, spec=spec or BudgetSpec())
            ledger.experiments = ledger.spec.max_experiments
            ledger.student_calls = ledger.spec.max_student_calls
            return ledger
        stored_spec = BudgetSpec.from_mapping(raw.get("spec"))
        return cls(
            path=path,
            spec=spec or stored_spec,
            experiments=int(raw.get("experiments", 0)),
            debug_runs=int(raw.get("debug_runs", 0)),
            student_calls=int(raw.get("student_calls", 0)),
            reveals=int(raw.get("reveals", 0)),
            probe_calls=int(raw.get("probe_calls", 0)),
            revealed_ids=list(raw.get("revealed_ids", [])),
            runs=list(raw.get("runs", [])),
            best_run_id=raw.get("best_run_id"),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec": asdict(self.spec),
            "experiments": self.experiments,
            "debug_runs": self.debug_runs,
            "student_calls": self.student_calls,
            "reveals": self.reveals,
            "probe_calls": self.probe_calls,
            "revealed_ids": self.revealed_ids,
            "runs": self.runs,
            "best_run_id": self.best_run_id,
        }
        handle, temporary = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".ledger-", suffix=".json"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    # -- reservation ------------------------------------------------------

    def reserve(
        self,
        *,
        experiments: int = 0,
        debug_runs: int = 0,
        calls: int = 0,
        reveals: int = 0,
        probe_calls: int = 0,
    ) -> None:
        """Charge a batch of resources, or raise without charging any of them."""
        checks = (
            (self.experiments + experiments, self.spec.max_experiments, "experiment"),
            (self.debug_runs + debug_runs, self.spec.max_debug_runs, "dry-run"),
            (self.student_calls + calls, self.spec.max_student_calls, "student call"),
            (self.reveals + reveals, self.spec.max_reveals, "train reveal"),
            (self.probe_calls + probe_calls, self.spec.max_probe_calls, "probe call"),
        )
        for proposed, limit, label in checks:
            if proposed > limit:
                raise BudgetExhausted(
                    f"{label} budget exhausted ({limit} allowed). {self.advice()}"
                )
        self.experiments += experiments
        self.debug_runs += debug_runs
        self.student_calls += calls
        self.reveals += reveals
        self.probe_calls += probe_calls
        self.save()

    def refund(self, *, calls: int = 0, experiments: int = 0, debug_runs: int = 0) -> None:
        """Return resources reserved for a run that never reached the student."""
        self.student_calls = max(0, self.student_calls - calls)
        self.experiments = max(0, self.experiments - experiments)
        self.debug_runs = max(0, self.debug_runs - debug_runs)
        self.save()

    def record_run(self, record: RunRecord) -> None:
        """Append a run and update ``best_run_id`` when it improves on the best."""
        self.runs.append(asdict(record))
        if record.kind == "experiment" and record.delta is not None:
            best = self.best_run()
            if best is None or record.delta > (best.get("delta") or float("-inf")):
                self.best_run_id = record.run_id
        self.save()

    def best_run(self) -> dict[str, Any] | None:
        if self.best_run_id is None:
            return None
        return next(
            (run for run in self.runs if run.get("run_id") == self.best_run_id), None
        )

    def mark_revealed(self, item_ids: list[str]) -> None:
        for item_id in item_ids:
            if item_id not in self.revealed_ids:
                self.revealed_ids.append(item_id)
        self.save()

    # -- reporting --------------------------------------------------------

    def remaining(self) -> dict[str, int]:
        return {
            "experiments": self.spec.max_experiments - self.experiments,
            "dry_runs": self.spec.max_debug_runs - self.debug_runs,
            "student_calls": self.spec.max_student_calls - self.student_calls,
            "reveals": self.spec.max_reveals - self.reveals,
            "probe_calls": self.spec.max_probe_calls - self.probe_calls,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": {
                "experiments": self.experiments,
                "dry_runs": self.debug_runs,
                "student_calls": self.student_calls,
                "reveals": self.reveals,
                "probe_calls": self.probe_calls,
            },
            "limits": asdict(self.spec),
            "remaining": self.remaining(),
            "runs_recorded": len(self.runs),
            "best_run_id": self.best_run_id,
            "questions_revealed": len(self.revealed_ids),
        }

    def fraction_left(self) -> float:
        """The tightest remaining budget, as a fraction. Drives the submit nag."""
        limits = (
            (self.spec.max_experiments - self.experiments, self.spec.max_experiments),
            (
                self.spec.max_student_calls - self.student_calls,
                self.spec.max_student_calls,
            ),
        )
        return min(
            (left / total if total else 1.0) for left, total in limits
        )

    def advice(self) -> str:
        """A short, actionable statement of what the remaining budget buys."""
        left = self.remaining()
        return (
            f"Remaining: {left['experiments']} evaluation(s), "
            f"{left['dry_runs']} dry run(s), {left['student_calls']} student call(s), "
            f"{left['reveals']} reveal(s)."
        )


class StateLedger:
    """Small Corral-native ledger backed by a mutable environment namespace.

    Corral commits the namespace after each trusted tool call, so this ledger does
    not need a workspace file or inter-process locking. The mapping is intentionally
    plain JSON-shaped data: it is copied into an ExecutionState by the environment
    and returned through ``ToolExecutionResult`` after every mutation.
    """

    def __init__(self, state: dict[str, Any], spec: BudgetSpec) -> None:
        self.state = state
        self.spec = spec
        state.setdefault("experiments", 0)
        state.setdefault("debug_runs", 0)
        state.setdefault("student_calls", 0)
        state.setdefault("reveals", 0)
        state.setdefault("probe_calls", 0)
        state.setdefault("revealed_ids", [])
        state.setdefault("runs", [])
        state.setdefault("best_run_id", None)

    @property
    def experiments(self) -> int:
        return int(self.state["experiments"])

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self.state["runs"]

    @property
    def revealed_ids(self) -> list[str]:
        return self.state["revealed_ids"]

    @property
    def best_run_id(self) -> str | None:
        return self.state.get("best_run_id")

    def reserve(
        self,
        *,
        experiments: int = 0,
        debug_runs: int = 0,
        calls: int = 0,
        reveals: int = 0,
        probe_calls: int = 0,
    ) -> None:
        proposed = {
            "experiments": self.experiments + experiments,
            "debug_runs": int(self.state["debug_runs"]) + debug_runs,
            "student_calls": int(self.state["student_calls"]) + calls,
            "reveals": int(self.state["reveals"]) + reveals,
            "probe_calls": int(self.state["probe_calls"]) + probe_calls,
        }
        limits = {
            "experiments": self.spec.max_experiments,
            "debug_runs": self.spec.max_debug_runs,
            "student_calls": self.spec.max_student_calls,
            "reveals": self.spec.max_reveals,
            "probe_calls": self.spec.max_probe_calls,
        }
        for name, value in proposed.items():
            if value > limits[name]:
                raise BudgetExhausted(
                    f"{name.replace('_', ' ')} budget exhausted ({limits[name]} allowed)"
                )
        self.state.update(proposed)

    def refund(self, *, calls: int = 0) -> None:
        self.state["student_calls"] = max(0, int(self.state["student_calls"]) - calls)

    def record_run(self, record: RunRecord) -> None:
        payload = asdict(record)
        self.runs.append(payload)
        if record.kind == "experiment" and record.delta is not None:
            best = self.best_run()
            if best is None or record.delta > (best.get("delta") or float("-inf")):
                self.state["best_run_id"] = record.run_id

    def best_run(self) -> dict[str, Any] | None:
        run_id = self.best_run_id
        if run_id is None:
            return None
        return next((run for run in self.runs if run.get("run_id") == run_id), None)

    def mark_revealed(self, item_ids: list[str]) -> None:
        for item_id in item_ids:
            if item_id not in self.revealed_ids:
                self.revealed_ids.append(item_id)

    def remaining(self) -> dict[str, int]:
        return {
            "experiments": self.spec.max_experiments - self.experiments,
            "dry_runs": self.spec.max_debug_runs - int(self.state["debug_runs"]),
            "student_calls": self.spec.max_student_calls - int(self.state["student_calls"]),
            "reveals": self.spec.max_reveals - int(self.state["reveals"]),
            "probe_calls": self.spec.max_probe_calls - int(self.state["probe_calls"]),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": {
                "experiments": self.experiments,
                "dry_runs": int(self.state["debug_runs"]),
                "student_calls": int(self.state["student_calls"]),
                "reveals": int(self.state["reveals"]),
                "probe_calls": int(self.state["probe_calls"]),
            },
            "limits": asdict(self.spec),
            "remaining": self.remaining(),
            "runs_recorded": len(self.runs),
            "best_run_id": self.best_run_id,
            "questions_revealed": len(self.revealed_ids),
        }

    def fraction_left(self) -> float:
        pairs = (
            (self.spec.max_experiments - self.experiments, self.spec.max_experiments),
            (
                self.spec.max_student_calls - int(self.state["student_calls"]),
                self.spec.max_student_calls,
            ),
        )
        return min(left / total if total else 1.0 for left, total in pairs)

    def advice(self) -> str:
        left = self.remaining()
        return (
            f"Remaining: {left['experiments']} evaluation(s), "
            f"{left['dry_runs']} dry run(s), {left['student_calls']} student call(s), "
            f"{left['reveals']} reveal(s)."
        )
