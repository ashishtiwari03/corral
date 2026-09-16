import json
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from corral_md import env, score
from modal.volume import FileEntryType

from corral.core.state import ExecutionState, RuntimeState
from corral.evaluation import TaskScorer


@pytest.fixture
def modal_references(monkeypatch):
    files = {
        "eval_structures": {"Si.data", "Al.data"},
        "potentials": {"SW/Si.sw", "EAM/Al99.eam.alloy", "BKS/pot.mod"},
    }

    def volume(name):
        return SimpleNamespace(
            listdir=lambda path: [SimpleNamespace(path=path, type=FileEntryType.FILE)]
            if path in files[name]
            else []
        )

    monkeypatch.setattr(score.modal.Volume, "from_name", volume)


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("level_1/tasks_json", 10),
        ("level_2/tasks_json", 10),
        ("level_1/subtasks_json", 14),
    ],
)
def test_all_shipped_tasks_load_from_unrelated_cwd(
    source, count, tmp_path, monkeypatch, modal_references
):
    monkeypatch.chdir(tmp_path)
    tasks = env.load_tasks_from_json(env.PACKAGE_DATA_ROOT / source, str(tmp_path))
    assert len(tasks) == count
    assert all(callable(task.scoring_fn) for task in tasks.values())


class _ReferenceModel:
    """Return known predictions to exercise reference loading, not model quality."""

    def __init__(self, reference):
        self.x = np.load(reference / "X_test.npy")
        self.y = np.load(reference / "y_test.npy")

    def predict(self, x):
        np.testing.assert_array_equal(x, self.x)
        return self.y


def test_l2_reference_scorers_work_after_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json", str(tmp_path)
    )
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    references = env.PACKAGE_DATA_ROOT / "ground_truth/level_2"

    def evaluate(number, answer):
        state = ExecutionState(
            through_commit_hash="a" * 64,
            execution_id="test",
            branch_id="main",
            runtime=RuntimeState(status="submitted"),
            submission=answer,
        )
        return (
            TaskScorer(tasks[f"level_2_task_{number}"], workspace=tmp_path)
            .evaluate(state)
            .score
        )

    phonon = json.loads((references / "task_5/phonon.json").read_text())
    assert evaluate(5, json.dumps(phonon)) == 1.0
    vdos = next((references / "task_6").glob("*.csv"))
    outputs = tmp_path / "arbitrary/nested/results"
    outputs.mkdir(parents=True)
    shutil.copyfile(vdos, outputs / "vdos.csv")
    assert evaluate(6, "arbitrary/nested/results/vdos.csv") == 1.0
    for number in (8, 9):
        model_path = outputs / f"model_{number}.pkl"
        model_path.write_bytes(
            pickle.dumps(_ReferenceModel(references / f"task_{number}"))
        )
        assert (
            evaluate(
                number, json.dumps({"model": str(model_path.relative_to(tmp_path))})
            )
            == 1.0
        )


@pytest.mark.parametrize(
    ("name", "params", "error"),
    [
        ("unknown_scorer", {}, "not found in the registry"),
        ("check_numerical", {}, "missing .* required"),
        ("check_numerical", {"target": 1}, "tolerance"),
        (
            "check_numerical",
            {"target": 1, "tolerence": 0.1},
            "unexpected keyword argument 'tolerence'",
        ),
        (
            "check_r2",
            {"hidden_test_path": "unused", "threshold": 0.9, "target": None},
            "unexpected keyword argument 'target'",
        ),
        (
            "check_phonon",
            {"target": "missing.json", "tolerance": 0.1},
            "Reference file not found",
        ),
        ("check_phonon", {"target": "", "tolerance": 0.1}, "Reference file not found"),
    ],
)
def test_invalid_tasks_fail_before_environments_are_built(
    name, params, error, tmp_path, monkeypatch
):
    source = tmp_path / "level_1/tasks_json"
    source.mkdir(parents=True)
    task_file = source / "invalid.json"
    task_file.write_text(
        json.dumps(
            [
                {
                    "id": "invalid_task",
                    "scoring_function": name,
                    "scoring_params": params,
                }
            ]
        )
    )
    monkeypatch.setattr(env, "PACKAGE_DATA_ROOT", tmp_path)

    def unexpected_build(*args, **kwargs):
        pytest.fail("Environments must not be built for invalid tasks")

    monkeypatch.setattr(env, "build_environments", unexpected_build)
    with pytest.raises(ValueError, match=error) as exc:
        env.create_environments(work_dir=str(tmp_path / "work"))
    assert "invalid_task" in str(exc.value)
    assert str(task_file) in str(exc.value)


@pytest.mark.parametrize("missing", ["X_test.npy", "y_test.npy"])
def test_r2_requires_both_reference_arrays(missing, tmp_path):
    for filename in {"X_test.npy", "y_test.npy"} - {missing}:
        np.save(tmp_path / filename, np.zeros(2))
    with pytest.raises(ValueError, match=missing):
        env.get_scoring_function(
            "check_r2", {"hidden_test_path": str(tmp_path), "threshold": 0.9}
        )


def test_cosine_requires_reference_csv_files(tmp_path):
    (tmp_path / "directory.csv").mkdir()
    with pytest.raises(ValueError, match="must contain CSV files"):
        env.get_scoring_function(
            "check_cosine_similarity", {"target": str(tmp_path), "threshold": 0.9}
        )


@pytest.mark.parametrize(
    ("name", "params"),
    [
        (
            "check_structure",
            {"target": "/eval_structures/missing.data", "atom_style": "full"},
        ),
        ("check_potential_file", {"target": "/potentials/SW/missing.sw"}),
    ],
)
def test_missing_modal_references_fail_at_load_time(name, params, modal_references):
    with pytest.raises(ValueError, match="Reference file not found"):
        env.get_scoring_function(name, params)


def test_reference_resolution_does_not_modify_params():
    params = {"target": "ground_truth/level_2/task_5/phonon.json", "tolerance": 0.1}
    original = params.copy()
    env.get_scoring_function("check_phonon", params)
    assert params == original
