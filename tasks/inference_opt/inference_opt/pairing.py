"""Reading run outcomes back out of inspect logs, and comparing them properly.

A raw accuracy delta is not enough to tell whether a policy did anything. Because
the baseline and the policy answer *the same* questions, the comparison is paired,
and the informative quantity is how many items changed hands:

* ``b`` — items the policy fixed,
* ``c`` — items the policy broke.

A delta of +0.10 with ``b + c == 3`` got lucky three times. The same delta with
``b + c == 15`` is a policy that is genuinely doing something and also causing harm.
The score alone cannot tell those apart, so every report carries ``n_changed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ItemOutcome",
    "RunOutcome",
    "compare",
    "mcnemar",
    "read_outcomes",
    "wilson_interval",
]


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    """What happened on one question."""

    item_id: str
    correct: bool
    answer: str = ""
    target: str = ""
    calls: int = 0
    error: str = ""
    unparseable: bool = False
    benchmark: str = ""
    category: str | None = None


@dataclass
class RunOutcome:
    """Every item of one run, plus the aggregates the tools report."""

    items: list[ItemOutcome] = field(default_factory=list)
    model_calls_logged: int = 0

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def n_correct(self) -> int:
        return sum(1 for item in self.items if item.correct)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def stderr(self) -> float:
        if self.n < 2:
            return 0.0
        p = self.accuracy
        return math.sqrt(max(0.0, p * (1 - p)) / self.n)

    @property
    def by_id(self) -> dict[str, ItemOutcome]:
        return {item.item_id: item for item in self.items}

    def by_category(self) -> dict[str, dict[str, float | int]]:
        buckets: dict[str, list[ItemOutcome]] = {}
        for item in self.items:
            buckets.setdefault(item.category or "(none)", []).append(item)
        return {
            name: {
                "n": len(group),
                "n_correct": sum(1 for item in group if item.correct),
                "accuracy": sum(1 for item in group if item.correct) / len(group),
            }
            for name, group in sorted(buckets.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "stderr": round(self.stderr, 4),
            "n_unparseable": sum(1 for item in self.items if item.unparseable),
            "n_errored": sum(1 for item in self.items if item.error),
            "calls_used": sum(item.calls for item in self.items),
        }


def _score_is_correct(value: Any) -> bool:
    """inspect scores are ``"C"``/``"I"``, or numeric for partial-credit scorers."""
    if isinstance(value, str):
        return value.upper() == "C"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 1.0
    return False


def read_outcomes(log_dir: Path | str, predictions: list[dict[str, Any]] | None = None) -> RunOutcome:
    """Collect per-item outcomes from every inspect log under ``log_dir``.

    ``predictions`` (the eval host's own record) supplies call counts and error
    text; the log supplies the authoritative correctness. Reading both lets the
    caller cross-check the host's reported spend against the model events inspect
    recorded independently.
    """
    from inspect_ai.log import read_eval_log

    log_dir = Path(log_dir)
    extra = {row["item_id"]: row for row in (predictions or [])}
    items: list[ItemOutcome] = []
    logged_calls = 0

    for path in sorted(log_dir.glob("*.json")) + sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            scores = sample.scores or {}
            # A task has exactly one scorer here, so the first score is the score.
            correct = any(_score_is_correct(score.value) for score in scores.values())
            policy_meta = (sample.metadata or {}).get("policy", {}) or {}
            side = extra.get(str(sample.id), {})
            target = sample.target
            items.append(
                ItemOutcome(
                    item_id=str(sample.id),
                    correct=correct,
                    answer=str(policy_meta.get("answer", "")),
                    target=", ".join(target) if isinstance(target, list) else str(target),
                    calls=int(side.get("calls", policy_meta.get("calls", 0)) or 0),
                    error=str(side.get("error", policy_meta.get("error", "")) or ""),
                    unparseable=bool(policy_meta.get("unparseable", False)),
                    benchmark=str((sample.metadata or {}).get("benchmark", "")),
                    category=(sample.metadata or {}).get("category"),
                )
            )
            for usage in (sample.model_usage or {}).values():
                # One ModelEvent per request; inspect aggregates them per sample.
                logged_calls += 1 if usage else 0

    return RunOutcome(items=items, model_calls_logged=logged_calls)


def mcnemar(
    baseline: dict[str, bool], policy: dict[str, bool]
) -> tuple[int, int, float]:
    """Return ``(b, c, p)`` over the items both runs attempted.

    ``b`` = fixed by the policy, ``c`` = broken by it. ``p`` is the exact binomial
    two-sided test, which is the right one at these sample sizes — the chi-square
    approximation is unreliable when ``b + c`` is small, and here it usually is.
    """
    shared = set(baseline) & set(policy)
    b = sum(1 for key in shared if policy[key] and not baseline[key])
    c = sum(1 for key in shared if baseline[key] and not policy[key])
    n = b + c
    if n == 0:
        return 0, 0, 1.0
    # Exact two-sided binomial test against p=0.5.
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return b, c, min(1.0, 2 * tail)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly near 0 and 1, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class Comparison:
    """A policy run measured against the baseline on the same questions."""

    accuracy: float
    baseline: float
    delta: float
    n: int
    n_changed: int
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float
    delta_stderr: float
    calls_used: int = 0

    @property
    def meaningful(self) -> bool:
        """Whether the change is larger than roughly two standard errors."""
        return abs(self.delta) > 2 * self.delta_stderr and self.n_changed > 0

    def verdict(self) -> str:
        """One plain sentence, so a score is never read as a point estimate."""
        if self.n_changed == 0:
            return "No question changed outcome: this policy did nothing measurable."
        direction = (
            "better"
            if self.delta > 0
            else "worse"
            if self.delta < 0
            else "no net change"
        )
        confidence = (
            "larger than noise" if self.meaningful else "within noise on this many items"
        )
        return (
            f"{self.delta:+.3f} ({direction}), {self.n_changed} question(s) changed "
            f"({self.mcnemar_b} fixed, {self.mcnemar_c} broken), p={self.mcnemar_p:.3f} "
            f"- {confidence}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "baseline": round(self.baseline, 4),
            "delta": round(self.delta, 4),
            "delta_stderr": round(self.delta_stderr, 4),
            "n": self.n,
            "n_changed": self.n_changed,
            "mcnemar_b": self.mcnemar_b,
            "mcnemar_c": self.mcnemar_c,
            "mcnemar_p": round(self.mcnemar_p, 4),
            "meaningful": self.meaningful,
            "calls_used": self.calls_used,
            "verdict": self.verdict(),
        }


def compare(policy: RunOutcome, baseline: dict[str, bool]) -> Comparison:
    """Compare a run against per-item baseline correctness on the same questions."""
    policy_map = {item.item_id: item.correct for item in policy.items}
    b, c, p = mcnemar(baseline, policy_map)
    shared = set(baseline) & set(policy_map)
    n = len(shared) or policy.n
    baseline_accuracy = (
        sum(1 for key in shared if baseline[key]) / len(shared) if shared else 0.0
    )
    # Paired standard error: only the discordant pairs carry information.
    delta_stderr = math.sqrt(b + c) / n if n else 0.0
    return Comparison(
        accuracy=policy.accuracy,
        baseline=baseline_accuracy,
        delta=policy.accuracy - baseline_accuracy,
        n=n,
        n_changed=b + c,
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_p=p,
        delta_stderr=delta_stderr,
        calls_used=sum(item.calls for item in policy.items),
    )
