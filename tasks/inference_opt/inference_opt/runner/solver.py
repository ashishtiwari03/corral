"""The inspect solver that delegates to a teacher-written policy.

Inspect normally owns the prompting: a solver calls ``generate`` and the model answers.
Here the *policy* owns all prompting, so this solver never calls ``generate`` at all.
It is a pure adapter — hand the policy a :class:`~inference_opt.api.Question` and a metered client, take back an answer, and write that answer into ``state.output`` in the canonical form inspect's scorers expect.

That last step is what makes ``choice()`` work: it grades from the correctness flags on ``state.choices``, which the ``multiple_choice`` solver would normally set.
Since we replaced that solver, we set them ourselves from the policy's answer.
"""

from __future__ import annotations

import inspect as _inspect
import time
import traceback
from typing import TYPE_CHECKING, Any

import anyio
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.solver._multiple_choice import (
    parse_answers,
    set_choices_based_on_generated_response,
)

from inference_opt.api import Answer, BudgetExhausted, Question
from inference_opt.scoring_specs import NO_ANSWER, canonical_completion

if TYPE_CHECKING:
    from pathlib import Path

    from inference_opt.policy import LoadedPolicy
    from inference_opt.runner.runtime import RunRuntime

__all__ = ["policy_solver", "question_from_state", "unwrap_answer"]


def question_from_state(state: TaskState) -> Question:
    """Rebuild the policy-facing question from an inspect sample.

    Reads only what the controller put in ``metadata``; ``state.target`` is never touched, and is empty on train runs regardless.
    """
    metadata = state.metadata or {}
    choices = tuple(choice.value for choice in state.choices) if state.choices else None
    labels = (
        tuple(chr(ord("A") + index) for index in range(len(choices)))
        if choices
        else None
    )
    return Question(
        id=str(state.sample_id),
        text=str(metadata.get("question") or state.input_text),
        benchmark=str(metadata.get("benchmark", "")),
        answer_type=str(metadata.get("answer_type", "text")),  # type: ignore[arg-type]
        choices=choices,
        choice_labels=labels,
        topic=metadata.get("category"),
        index=int(metadata.get("index", 0)),
        total=int(metadata.get("total", 1)),
    )


def unwrap_answer(raw: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Accept either a bare value or a structured :class:`Answer`."""
    if isinstance(raw, Answer):
        return raw.final, [dict(entry) for entry in raw.trace]
    return raw, []


async def _invoke(fn: Any, question: Question, context: Any) -> Any:
    """Run a policy method, whether it is sync or async.

    ``anyio.to_thread.run_sync`` copies contextvars into the worker thread and
    ``anyio.from_thread.run`` copies them back, so model calls the policy makes on
    that thread are still attributed to the right sample in the inspect transcript.
    """
    if _inspect.iscoroutinefunction(fn):
        return await fn(question, context)

    def call() -> Any:
        return fn(question, context)

    return await anyio.to_thread.run_sync(call)


@solver
def policy_solver(policy: LoadedPolicy, runtime: RunRuntime, artifacts: Path) -> Solver:
    """Delegate one sample to the policy and record what happened."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate  # the policy does its own generating, by design
        question = question_from_state(state)
        meter = runtime.meter_for(question.id)
        context, log_lines = runtime.make_solve_context(meter, artifacts)
        started = time.monotonic()
        error = ""
        raw: Any = ""

        try:
            raw = await _invoke(policy.solve, question, context)
        except BudgetExhausted as exc:
            error = f"budget_exhausted: {exc}"
            runtime.note_exhaustion(question)
            # A policy that left a cheap guess behind still scores on this item.
            raw = context.scratch.get("fallback", "")
        except Exception as exc:  # a policy fault costs this item, not the run
            error = f"{type(exc).__name__}: {exc}"
            state.metadata["policy_traceback"] = traceback.format_exc(limit=8)
            raw = context.scratch.get("fallback", "")

        final, trace = unwrap_answer(raw)
        answer_format = str((state.metadata or {}).get("answer_format", "text"))
        completion = canonical_completion(
            final,
            answer_format=answer_format,
            n_choices=len(state.choices) if state.choices else 0,
            choices=question.choices,
        )

        state.output = ModelOutput.from_content(model="policy", content=completion)
        state.messages.append(ChatMessageAssistant(content=completion))

        # choice() grades from these flags, not from the completion text.
        if answer_format.startswith("mcq") and state.choices:
            set_choices_based_on_generated_response(
                state,
                parse_answers(state, multiple_correct=answer_format == "mcq_multi"),
            )

        seconds = time.monotonic() - started
        state.metadata["policy"] = {
            "answer": completion,
            "raw_answer": str(final)[:2000],
            "calls": meter.used,
            "output_tokens": meter.tokens,
            "seconds": round(seconds, 3),
            "components": meter.by_component,
            "log": log_lines[:50],
            "trace": trace[:10],
            "error": error,
            "unparseable": completion.endswith(NO_ANSWER),
        }
        runtime.emit_prediction(question, completion, meter, error, log_lines, seconds)
        return state

    return solve
