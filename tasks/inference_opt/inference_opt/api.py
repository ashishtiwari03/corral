"""The contract between the harness and a teacher-written inference policy.

A policy is duck-typed: it imports nothing from this module. These types exist so
the harness, the tests and the generated documentation all agree on one shape, and
so the AST validator (:mod:`inference_opt.validator`) can keep its import allowlist
maximally strict — a policy that had to ``import inference_opt`` would need the
package inside the sandbox, and the allowlist would have to admit it.

A policy supplies, in ``policy.py``:

* a ``Policy`` class with ``solve(question, ctx)`` and optionally ``setup(ctx)``, or
* a module-level ``solve(question, model_client, context)`` (the legacy form), which
  :func:`inference_opt.policy.discover_policy` wraps automatically.

Optionally a module- or class-level ``MANIFEST`` dict declares how the policy wants
to be run; see :class:`PolicyManifest`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

__all__ = [
    "Answer",
    "Answerable",
    "AnswerType",
    "BudgetExhausted",
    "ComponentSpec",
    "ExecutionMode",
    "LabeledExample",
    "Memory",
    "Message",
    "MemoryMode",
    "PolicyManifest",
    "Prompt",
    "Question",
    "SetupContext",
    "SolveContext",
    "StudentClient",
]

#: What a policy may return as its final answer. Anything else is coerced with ``str``.
Answerable = str | float | int

#: One chat message, e.g. ``{"role": "user", "content": "..."}``.
Message = Mapping[str, str]

#: What every client method accepts: a bare prompt or an explicit message list.
Prompt = str | Sequence[Message]

AnswerType = Literal["mcq", "numeric", "text"]
ExecutionMode = Literal["sequential", "parallel"]
MemoryMode = Literal["none", "shared"]


class BudgetExhausted(RuntimeError):
    """Raised by :meth:`StudentClient.generate` when no student calls remain.

    Derives from ``RuntimeError`` so a policy written defensively as
    ``except Exception: return fallback`` still produces an answer, and so the
    scaffold's original ``RuntimeError("... budget exhausted")`` contract holds.

    Remaining questions in a run are still attempted after this is raised; every
    further call raises immediately. A policy that catches it and returns a cheap
    guess scores better than one that does not, which is the intended incentive.
    """


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question, with the answer withheld.

    ``index`` and ``total`` let a policy pace itself across a run — for example,
    spending more of a shared budget early and falling back to single-shot late.
    """

    id: str
    text: str
    benchmark: str
    answer_type: AnswerType
    choices: tuple[str, ...] | None = None
    choice_labels: tuple[str, ...] | None = None
    topic: str | None = None
    index: int = 0
    total: int = 1

    @property
    def is_multiple_choice(self) -> bool:
        return self.answer_type == "mcq" and bool(self.choices)

    def rendered_choices(self) -> str:
        """Return ``"A) first\\nB) second"``, or an empty string when free-form."""
        if not self.choices:
            return ""
        labels = self.choice_labels or tuple(
            chr(ord("A") + position) for position in range(len(self.choices))
        )
        return "\n".join(
            f"{label}) {choice}"
            for label, choice in zip(labels, self.choices, strict=False)
        )


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """A train question the teacher agent unlocked, with its gold answer.

    ``baseline_completion`` and ``baseline_correct`` describe what the student said
    zero-shot, so a policy can select demonstrations by the student's actual failure
    modes rather than by guesswork. Only ever produced from the train split.
    """

    question: Question
    answer: str
    baseline_completion: str = ""
    baseline_correct: bool = False


@dataclass(frozen=True, slots=True)
class Answer:
    """A policy's structured reply. Returning a bare string is equally valid."""

    final: Answerable
    confidence: float | None = None
    rationale: str | None = None
    trace: tuple[Mapping[str, Any], ...] = ()


@runtime_checkable
class StudentClient(Protocol):
    """The only permitted access to the student model.

    A policy that reaches any other model — by import, HTTP, or subprocess — is
    rejected by the validator and scores zero. Every method charges the run's call
    budget before issuing a request, and raises :class:`BudgetExhausted` when the
    reservation fails.
    """

    def generate(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        component: str | None = None,
    ) -> str:
        """Complete one prompt. Charges one call."""
        ...

    def sample(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> list[str]:
        """Draw ``n`` completions of one prompt. Charges ``n`` calls."""
        ...

    def batch(
        self,
        prompts: Sequence[Prompt],
        **kwargs: Any,
    ) -> list[str]:
        """Complete several prompts concurrently. Charges ``len(prompts)`` calls."""
        ...

    @property
    def calls_used(self) -> int: ...

    @property
    def calls_remaining(self) -> int: ...


@runtime_checkable
class Memory(Protocol):
    """Mutable state shared across every question of one run.

    Only available when the manifest declares ``memory="shared"``, which forces
    sequential execution — concurrent questions mutating shared state is a
    nondeterminism trap, so the two are deliberately not combinable. With
    ``memory="none"`` the mutating methods raise.
    """

    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def append(self, key: str, value: Any) -> None: ...
    def items(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Passed to the optional ``Policy.setup``, once, before any question.

    This is where prompt optimization and few-shot selection belong: it is the only
    place a policy sees gold answers. Calls made here are charged against the
    separate ``setup_calls`` allowance declared in the manifest, so setup cannot eat
    the per-question reserves.
    """

    student: StudentClient
    train_examples: tuple[LabeledExample, ...]
    memory: Memory
    artifacts: Path
    budget_remaining: int
    log: Callable[[str], None]
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolveContext:
    """Passed to ``Policy.solve`` for each question.

    ``question_budget`` is what this question may still spend; ``budget_remaining``
    is what the whole run has left. An adaptive policy should consult both rather
    than assuming a fixed per-question allowance.

    Anything written to ``scratch["fallback"]`` is used as the answer if the policy
    later raises — the cheapest way to stay scored under budget exhaustion.
    """

    student: StudentClient
    memory: Memory
    artifacts: Path
    budget_remaining: int
    question_budget: int
    log: Callable[[str], None]
    scratch: dict[str, Any] = field(default_factory=dict)
    #: Tags every call made inside it, so diagnostics can attribute spend and
    #: accuracy per component. ``None`` when the runtime supplies no tagging.
    component: Callable[[str], AbstractContextManager[None]] | None = None


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One named part of a multi-component policy, for per-component diagnostics."""

    name: str
    kind: str = "other"
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """How a policy wants to be run. Every field has a working default.

    Declared by the policy as a plain ``MANIFEST`` dict; unknown keys are rejected
    so a typo surfaces at ``dry_run_policy`` rather than silently doing nothing.
    """

    name: str = "policy"
    version: int = 1
    execution: ExecutionMode = "parallel"
    memory: MemoryMode = "none"
    max_calls_per_question: int = 8
    setup_calls: int = 0
    max_tokens_per_call: int = 2048
    components: tuple[ComponentSpec, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_calls_per_question < 1:
            raise ValueError("max_calls_per_question must be at least 1")
        if self.setup_calls < 0:
            raise ValueError("setup_calls cannot be negative")
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be at least 1")

    @property
    def effective_execution(self) -> ExecutionMode:
        """Shared memory forces sequential execution, whatever was declared."""
        return "sequential" if self.memory == "shared" else self.execution
