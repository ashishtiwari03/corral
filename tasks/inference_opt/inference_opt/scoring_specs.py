"""Turning a policy's free-form answer into something inspect can grade.

A policy may return anything: ``"B"``, ``"The answer is B."``, ``42``, a full chain
of thought ending in ``\\boxed{42}``. inspect's scorers expect a canonical shape —
``choice()`` reads the correctness flags that the ``multiple_choice`` solver sets on
``state.choices``, and ``match(numeric=True)`` looks for a value at the end of the
completion.

So this module does exactly one thing: it extracts the intended answer and rewrites
it as ``ANSWER: <value>``, which is the form inspect's own machinery understands.
Grading itself is then done by inspect's battle-tested scorers rather than by
anything hand-rolled here, because a silently wrong grader shows up as "the agent
improved nothing" and costs days of blaming the agent.

The one place a custom scorer is unavoidable is chembench's free numeric items,
whose upstream scorer is tolerance-based — exact string match would understate
every model including the baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    choice,
    match,
    scorer,
    stderr,
)

if TYPE_CHECKING:
    from inspect_ai.solver import TaskState

__all__ = [
    "ANSWER_PREFIX",
    "NO_ANSWER",
    "BenchmarkSpec",
    "canonical_completion",
    "extract_mcq_letters",
    "extract_numeric",
    "numeric_tolerance",
    "scorer_for",
    "spec_for",
]

ANSWER_PREFIX = "ANSWER:"

#: Emitted when nothing answer-shaped could be found. Scores incorrect, and is
#: counted separately so an agent can tell "wrong" from "unparseable".
NO_ANSWER = "NOANSWER"

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Ordered from most to least explicit. The first pattern that matches wins, so a
# model that both reasons aloud and states a final answer is read correctly.
_MCQ_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[ANSWER\]\s*([A-Za-z][A-Za-z,\s]*?)\s*\[/ANSWER\]", re.IGNORECASE),
    re.compile(r"(?im)^\s*ANSWER\s*[:=]\s*([A-Za-z][A-Za-z,\s]*?)\s*\.?\s*$"),
    re.compile(r"(?i)\bANSWER\s*[:=]\s*\(?([A-Za-z])\)?\b"),
    re.compile(r"(?i)\\boxed\{\s*\(?([A-Za-z])\)?\s*\}"),
    re.compile(r"(?i)\b(?:final answer|the answer)\s+is\s*:?\s*\(?([A-Za-z])\)?\b"),
    re.compile(r"(?i)\boption\s+\(?([A-Za-z])\)?\b"),
)

_NUMERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[ANSWER\]\s*(-?[\d.,/eE+\-]+)\s*\[/ANSWER\]", re.IGNORECASE),
    re.compile(r"(?im)^\s*ANSWER\s*[:=]\s*(-?[\d.,/eE+\-]+)\s*\.?\s*$"),
    re.compile(r"(?i)ANSWER\s*[:=]\s*\$?(-?[\d.,/eE+\-]+)"),
    re.compile(r"(?i)\\boxed\{\s*\$?(-?[\d.,/eE+\-]+)\s*\}"),
    re.compile(r"####\s*\$?(-?[\d.,/eE+\-]+)"),
    re.compile(r"(?i)\b(?:final answer|the answer)\s+is\s*:?\s*\$?(-?[\d.,/eE+\-]+)"),
)

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def _clean_number(text: str) -> str:
    """Strip separators and trailing punctuation from a captured number."""
    cleaned = text.strip().rstrip(".").replace(",", "").replace("$", "")
    return cleaned


def extract_numeric(text: str) -> str | None:
    """Pull the intended numeric answer out of a completion.

    Falls back to the last number in the text, which is what a model that simply
    ends its working with the result produces.
    """
    if not text:
        return None
    for pattern in _NUMERIC_PATTERNS:
        found = pattern.search(text)
        if found:
            cleaned = _clean_number(found.group(1))
            if _NUMBER.fullmatch(cleaned):
                return cleaned
    trailing = _NUMBER.findall(text)
    if trailing:
        return _clean_number(trailing[-1])
    return None


def extract_mcq_letters(
    text: str,
    *,
    n_choices: int,
    multi: bool = False,
    choices: tuple[str, ...] | None = None,
) -> list[str]:
    """Pull the intended choice letter(s) out of a completion.

    The ladder ends by matching the completion against the choice *text*, because a
    small model often answers ``"Paris"`` rather than ``"A"`` and scoring that
    incorrect would measure formatting compliance rather than knowledge.
    """
    if not text:
        return []
    valid = set(_LETTERS[:n_choices])

    def distinct_letters(raw: str) -> list[str]:
        letters = [
            character.upper()
            for character in re.sub(r"[,\s]+", "", raw)
            if character.upper() in valid
        ]
        return sorted(dict.fromkeys(letters))

    def resolve(found: list[str]) -> list[str] | None:
        """``None`` means "this match told us nothing, keep looking"."""
        if not found:
            return None
        if multi:
            return found
        # A single-answer question answered with two letters has not been
        # answered. Taking the first would hand a hedging policy a coin flip,
        # which a self-consistency strategy would learn to exploit on tied votes.
        return found if len(found) == 1 else []

    for pattern in _MCQ_PATTERNS:
        matched = pattern.search(text)
        if matched is None:
            continue
        # A pattern can fire on prose ("the answer is Paris" captures "P"), which
        # yields no valid letter and must not stop the ladder. An explicit but
        # ambiguous match ("ANSWER: A, C") must stop it, or a looser pattern
        # below would pick a single letter out of a hedged answer.
        resolved = resolve(distinct_letters(matched.group(1)))
        if resolved is not None:
            return resolved

    # A bare answer: the whole completion is just "B", or "B, D" for multi-select.
    # Checked token by token, because any looser rule reads prose as letters —
    # "The capital is Berlin" is nothing but letters and spaces.
    stripped = text.strip().strip(".()[]{}")
    tokens = [token for token in re.split(r"[,\s]+", stripped) if token]
    bare = bool(tokens) and all(
        len(token) == 1 and token.upper() in valid for token in tokens
    )
    # Multi-select may also run its letters together, as in "ABD".
    if not bare and multi and len(tokens) == 1:
        single = tokens[0].upper()
        bare = 1 < len(single) <= n_choices and set(single) <= valid
    if bare:
        resolved = resolve(distinct_letters(stripped))
        if resolved is not None:
            return resolved

    # Last resort: the model named the option instead of lettering it.
    if choices:
        lowered = text.strip().lower()
        for index, option in enumerate(choices):
            option_text = option.strip().lower()
            if option_text and (lowered == option_text or lowered.endswith(option_text)):
                return [_LETTERS[index]]
    return []


def canonical_completion(raw: object, *, answer_format: str, n_choices: int = 0,
                         choices: tuple[str, ...] | None = None) -> str:
    """Rewrite a policy's answer as ``ANSWER: <value>`` for inspect to grade.

    Returns ``ANSWER: NOANSWER`` when nothing answer-shaped was found, which scores
    incorrect — the same outcome as a wrong answer, but distinguishable in
    diagnostics so an agent can tell a formatting failure from a knowledge failure.
    """
    text = "" if raw is None else str(raw)
    # A policy returning megabytes must not be able to bloat the log or the scorer.
    text = text[:8192]

    if answer_format in ("mcq_single", "mcq_multi"):
        letters = extract_mcq_letters(
            text,
            n_choices=n_choices or len(choices or ()),
            multi=answer_format == "mcq_multi",
            choices=choices,
        )
        value = ", ".join(letters) if letters else NO_ANSWER
    elif answer_format == "numeric":
        value = extract_numeric(text) or NO_ANSWER
    else:
        value = text.strip().splitlines()[-1].strip() if text.strip() else NO_ANSWER

    return f"{ANSWER_PREFIX} {value}"


@scorer(metrics=[accuracy(), stderr()])
def numeric_tolerance(rtol: float = 0.01, atol: float = 1e-9) -> Scorer:
    """Score a numeric answer correct within a relative tolerance.

    chembench's own ``mae`` items are graded on tolerance upstream, so exact string
    comparison would systematically understate every model, baseline included, and
    make the improvement delta meaningless.
    """

    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_numeric(state.output.completion)
        expected = extract_numeric(target.text)
        if predicted is None or expected is None:
            return Score(
                value=INCORRECT,
                answer=predicted or NO_ANSWER,
                explanation=f"could not parse a number (target={target.text!r})",
            )
        try:
            got, want = float(predicted), float(expected)
        except ValueError:
            return Score(value=INCORRECT, answer=predicted, explanation="not a float")
        close = abs(got - want) <= max(atol, rtol * abs(want))
        return Score(
            value=CORRECT if close else INCORRECT,
            answer=predicted,
            explanation=f"{got} vs {want} (rtol={rtol})",
        )

    return score


def scorer_for(answer_format: str) -> Scorer:
    """The inspect scorer appropriate to one answer format."""
    if answer_format in ("mcq_single", "mcq_multi"):
        return choice()
    if answer_format == "numeric":
        return match(location="end", numeric=True)
    if answer_format == "numeric_tolerant":
        return numeric_tolerance()
    return match(location="end", ignore_case=True)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """How one (benchmark, answer_format) group is prompted and graded."""

    benchmark: str
    answer_format: str
    multiple_correct: bool = False

    @property
    def scorer_name(self) -> str:
        return {
            "mcq_single": "choice",
            "mcq_multi": "choice",
            "numeric": "match_numeric",
            "numeric_tolerant": "numeric_tolerance",
        }.get(self.answer_format, "match")

    def build_scorer(self) -> Scorer:
        return scorer_for(self.answer_format)


#: chembench numerics are tolerance-graded upstream; everything else numeric is exact.
_TOLERANT_NUMERIC = frozenset({"chembench"})


def spec_for(benchmark: str, answer_format: str) -> BenchmarkSpec:
    """Resolve the spec for one group of items."""
    resolved = answer_format
    if answer_format == "numeric" and benchmark in _TOLERANT_NUMERIC:
        resolved = "numeric_tolerant"
    return BenchmarkSpec(
        benchmark=benchmark,
        answer_format=resolved,
        multiple_correct=answer_format == "mcq_multi",
    )
