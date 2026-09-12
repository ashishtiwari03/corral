from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from uuid import uuid4

import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.env import (
    SUBMISSION_FORMAT,
    StargazerEnvironment,
    _configure_trial,
    _task_prompt,
    create_environments,
)
from stargazer.evaluator import make_stargazer_scorer
from stargazer.models import load_task
from stargazer.tools import create_tools

from corral.agents.session import AgentSession
from corral.core import Action, ActorRef, AgentStarted, CommitRequest
from corral.core.environment import Toolset
from corral.core.task import TaskDefinition
from corral.evaluation import TaskScorer
from corral.persistence import SQLiteCommitStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def environment(simple_task):
    task = TaskDefinition(
        name="Stargazer test",
        description="Infer the planetary system.",
        tools=["python_repl", "planet_from_fit", "evaluate_candidate"],
        scoring_fn=make_stargazer_scorer(simple_task),
        submission_format=SUBMISSION_FORMAT,
        scoring_inputs={"benchmark_task": simple_task, "max_evaluations": 2},
        prompt_fn=_task_prompt,
        setup_fn=_configure_trial,
        resolve_answer=False,
    )
    return StargazerEnvironment(
        simple_task.task_id,
        task,
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
    )


def _resume_session(environment, state, store):
    return AgentSession(
        environment,
        state,
        actor=ActorRef(kind="agent", actor_id="agent_0", run_id="agent"),
        runtime_actor=ActorRef(kind="runtime", actor_id="corral", run_id="runtime"),
        state_store=store,
        max_iterations=20,
    )


async def _start_session(environment, store):
    async def append(request_id, event):
        head = await store.head("main")
        await store.append(
            CommitRequest(
                request_id=request_id,
                branch_id="main",
                based_on_hash=head.hash if head else None,
                author=ActorRef(kind="runtime", actor_id="corral", run_id="runtime"),
                event=event,
            )
        )

    await append(
        "started",
        await asyncio.to_thread(
            environment.initial_event, execution_id=store.execution_id
        ),
    )
    await append(
        "configured",
        await asyncio.to_thread(environment.configure, await store.materialize("main")),
    )
    await append(
        "agent-started", AgentStarted(agent_run_id="agent", agent_id="agent_0")
    )
    return _resume_session(environment, await store.materialize("main"), store)


async def _execute(session, name, **arguments):
    result = await session.execute(
        Action(id=uuid4().hex, name=name, arguments=arguments, actor_id="agent_0")
    )
    assert result.success, result
    return result.result


def _evaluation_state(state):
    return state.environment.values["hidden_arguments"]["evaluation_session"]


def test_official_levels_have_fixed_reference_valid_synthetic_banks(tmp_path):
    levels = {
        level: create_environments(level=level, work_dir=tmp_path / f"level_{level}")
        for level in (1, 2)
    }

    assert {level: len(environments) for level, environments in levels.items()} == {
        1: 10,
        2: 10,
    }
    task_ids = [task_id for environments in levels.values() for task_id in environments]
    assert len(task_ids) == len(set(task_ids)) == 20

    expected_source_counts = {level: {"synthetic": 10} for level in (1, 2)}
    expected_difficulty_counts = {
        1: {5: 4, 6: 3, 7: 3},
        2: {8: 4, 9: 3, 10: 3},
    }
    expected_evaluation_budgets = {1: {4}, 2: {9}}
    for level, environments in levels.items():
        source_counts: dict[str, int] = {}
        difficulty_counts: dict[int, int] = {}
        for environment in environments.values():
            task = environment.current_task.scoring_inputs["benchmark_task"]
            source_counts[task.source] = source_counts.get(task.source, 0) + 1
            difficulty_counts[task.truth_difficulty] = (
                difficulty_counts.get(task.truth_difficulty, 0) + 1
            )
        assert source_counts == expected_source_counts[level]
        assert difficulty_counts == expected_difficulty_counts[level]
        assert {
            environment.current_task.scoring_inputs["max_evaluations"]
            for environment in environments.values()
        } == expected_evaluation_budgets[level]


@pytest.mark.parametrize("level", [3, "3"])
def test_removed_third_level_is_rejected(tmp_path, level):
    with pytest.raises(ValueError, match="level must be"):
        create_environments(level=level, work_dir=tmp_path)


def test_selected_zip_records_keep_exported_observations_and_truth(tmp_path):
    manifest = json.loads(
        (DEFAULT_DATA_ROOT / "selection_manifest.json").read_text(encoding="utf-8")
    )
    zip_ids = {
        1: {"seed15_diff5", "seed64_diff6", "seed43_diff7"},
        2: {
            "seed1_diff8",
            "seed17_diff8",
            "seed101_diff9",
            "seed93_diff9",
            "seed82_diff10",
        },
    }
    for level, expected in zip_ids.items():
        environments = create_environments(level=level, work_dir=tmp_path / str(level))
        rows = manifest["levels"][str(level)]["tasks"]
        assert {row["task_id"] for row in rows} == set(environments)
        assert {
            row["task_id"] for row in rows if row["selected_from"] == "zip"
        } == expected
        for task_id in expected:
            path = DEFAULT_DATA_ROOT / "synthetic" / f"{task_id}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            task = load_task(path, source="synthetic")
            # Exported RV data must bypass the legacy REBOUND transformation.
            assert asdict(task.observations) == {
                key: tuple(values) for key, values in raw["observations"].items()
            }
            assert task.star_mass_sun == raw["config"]["star"]["M_star_sun"]
            for planet, exported in zip(
                task.truth_planets, raw["config"]["planets"], strict=True
            ):
                assert {key: getattr(planet, key) for key in exported} == exported
            provenance = task.metadata["provenance"]
            assert provenance["source_task_id"] == task_id
            assert provenance["archive_sha256"] == manifest["archive"]["sha256"]
            answer = reference_submission(task, raw).model_dump_json()
            assert environments[task_id].current_task.scoring_fn(answer) == 1.0


def test_real_data_is_a_separate_challenge_split(tmp_path):
    environments = create_environments(level="real", work_dir=tmp_path / "real")

    assert len(environments) == 20
    assert {
        environment.current_task.scoring_inputs["benchmark_task"].source
        for environment in environments.values()
    } == {"real"}


@pytest.mark.anyio
async def test_released_task_loads_into_isolated_corral_environment(tmp_path):
    selector = tmp_path / "selector.json"
    selector.write_text(
        json.dumps(
            [
                {
                    "source": "synthetic",
                    "difficulty_min": 1,
                    "difficulty_max": 1,
                    "task_ids": ["seed300016_diff1"],
                    "max_evaluations": 2,
                }
            ]
        )
    )
    environments = create_environments(
        level=1, selector_path=selector, work_dir=tmp_path / "work"
    )

    assert list(environments) == ["seed300016_diff1"]
    environment = environments["seed300016_diff1"]
    assert set(environment.tools) == {
        "planet_from_fit",
        "python_repl",
        "evaluate_candidate",
    }

    benchmark_task = environment.current_task.scoring_inputs["benchmark_task"]
    truth_period = str(benchmark_task.truth_planets[0].P_days)
    async with SQLiteCommitStore(tmp_path / "released.sqlite3", "released") as store:
        session = await _start_session(environment.for_task(store.execution_id), store)
        state = await store.materialize("main")
        prompt = environment.get_task_prompt(state)
        assert truth_period not in prompt
        assert "times_days" in prompt
        assert '"noise_jitter_ms"' in prompt
        assert "Only the final Corral answer is scored" in prompt
        assert "Lomb" not in prompt
        hidden = state.environment.values["hidden_arguments"]
        assert hidden["benchmark_task"] == benchmark_task.task_id
        assert hidden["analysis_session"] is None
        assert "truth_planets" not in state.model_dump_json()

        planet = asdict(benchmark_task.truth_planets[0])
        planet.pop("m_true_mjup")
        answer = json.dumps({"planets": [planet], "noise_jitter_ms": 0.0})
        await _execute(session, "submit_answer", answer=answer)
        submitted = await store.materialize("main")
        assert environment.get_task_output(submitted).output["answer"] == answer
        assert TaskScorer(environment.current_task).evaluate(submitted).score == 1.0


@pytest.mark.anyio
async def test_same_task_trials_keep_independent_evaluation_state(
    tmp_path, environment, exact_submission
):
    async with (
        SQLiteCommitStore(tmp_path / "trials.sqlite3", "first") as first_store,
        SQLiteCommitStore(tmp_path / "trials.sqlite3", "second") as second_store,
    ):
        first = await _start_session(environment.for_task("first"), first_store)
        second = await _start_session(environment.for_task("second"), second_store)
        initial = await first_store.materialize("main")
        first_feedback = json.loads(
            await _execute(first, "evaluate_candidate", **exact_submission)
        )
        assert first_feedback["success"]
        assert not _evaluation_state(initial)["evaluations"]
        assert not _evaluation_state(await second_store.materialize("main"))[
            "evaluations"
        ]
        second_feedback = json.loads(
            await _execute(second, "evaluate_candidate", **exact_submission)
        )
        assert second_feedback["accepted"]
        assert second_feedback["evaluation_number"] == 1


@pytest.mark.anyio
async def test_evaluation_budget_and_success_lock_survive_fork_and_restore(
    tmp_path, environment, exact_submission
):
    database = tmp_path / "evaluations.sqlite3"
    async with SQLiteCommitStore(database, "evaluations") as store:
        session = await _start_session(environment, store)
        invalid = [{**exact_submission["planets"][0], "P_days": -1.0}]
        rejected = json.loads(
            await _execute(session, "evaluate_candidate", planets=invalid)
        )
        assert not rejected["accepted"]
        assert rejected["remaining_evaluations"] == 2
        assert not _evaluation_state(await store.materialize("main"))["evaluations"]

        feedback = json.loads(await _execute(session, "evaluate_candidate", planets=[]))
        assert feedback["accepted"]
        assert not feedback["success"]
        assert feedback["remaining_evaluations"] == 1
        branch = await session.fork_branch(branch_id="alternative")
        passed = json.loads(
            await _execute(session, "evaluate_candidate", **exact_submission)
        )
        assert passed["success"]

        alternate = json.loads(await _execute(branch, "evaluate_candidate", planets=[]))
        assert alternate["accepted"]
        assert not alternate["success"]
        exhausted = json.loads(
            await _execute(branch, "evaluate_candidate", **exact_submission)
        )
        assert not exhausted["accepted"]
        assert "exhausted" in exhausted["error"]
        committed = await store.materialize("main")
        assert _evaluation_state(committed)["locked"]
        assert not _evaluation_state(await store.materialize("alternative"))["locked"]

    async with SQLiteCommitStore(database, "evaluations") as restored_store:
        checkpoint = await restored_store.materialize("main")
        assert checkpoint == committed
        restored = _resume_session(
            environment.for_task("evaluations"), checkpoint, restored_store
        )
        locked = json.loads(
            await _execute(restored, "evaluate_candidate", **exact_submission)
        )
        assert not locked["accepted"]
        assert "locked" in locked["error"]
        assert locked["remaining_evaluations"] == 0
        assert _evaluation_state(
            await restored_store.materialize("main")
        ) == _evaluation_state(checkpoint)


@pytest.mark.anyio
async def test_python_state_restores_functions_arrays_and_isolated_branches(
    tmp_path, environment
):
    database = tmp_path / "analysis.sqlite3"
    async with SQLiteCommitStore(database, "analysis") as store:
        session = await _start_session(environment, store)
        initialized = await _execute(
            session,
            "python_repl",
            code="values = np.arange(3.0)\noffset = 2.0\ndef shifted():\n    return values + offset",
        )
        assert "successfully" in initialized
        committed = await store.materialize("main")
        assert isinstance(
            committed.environment.values["hidden_arguments"]["analysis_session"], str
        )
        branch = await session.fork_branch(branch_id="alternative")
        mutated = await _execute(
            branch,
            "python_repl",
            code="values[0] = 40.0\noffset = 3.0\nshifted().tolist()",
        )
        assert json.loads(mutated) == [43.0, 4.0, 5.0]
        assert (await store.materialize("main")).environment == committed.environment

    async with SQLiteCommitStore(database, "analysis") as restored_store:
        checkpoint = await restored_store.materialize("main")
        restored = _resume_session(
            environment.for_task("analysis"), checkpoint, restored_store
        )
        result = await _execute(restored, "python_repl", code="shifted().tolist()")
        assert json.loads(result) == [2.0, 3.0, 4.0]


@pytest.mark.anyio
@pytest.mark.parametrize("submit_correct_answer", [True, False])
async def test_only_the_final_submission_is_scored(
    tmp_path, environment, exact_submission, submit_correct_answer
):
    async with SQLiteCommitStore(tmp_path / "scoring.sqlite3", "scoring") as store:
        session = await _start_session(environment, store)
        feedback = json.loads(
            await _execute(session, "evaluate_candidate", **exact_submission)
        )
        assert feedback["success"]
        diagnostic_state = await store.materialize("main")
        assert diagnostic_state.submission is None
        assert environment.get_task_output(diagnostic_state) is None
        scorer = TaskScorer(environment.current_task)
        with pytest.raises(ValueError, match="completed"):
            scorer.evaluate(diagnostic_state)

        answer = json.dumps(exact_submission) if submit_correct_answer else "{}"
        await _execute(session, "submit_answer", answer=answer)
        submitted = await store.materialize("main")
        assert scorer.evaluate(submitted).score == float(submit_correct_answer)
