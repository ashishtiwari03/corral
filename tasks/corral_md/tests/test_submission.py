import asyncio
import json
import shutil

import pytest
from corral_md.env import PACKAGE_DATA_ROOT, load_tasks_from_json
from corral_md.submission import resolve_submission

from corral.core.state import ExecutionState, RuntimeState
from corral.evaluation import TaskScorer
from corral.persistence.workspace import WorkspaceManager


def evaluate(task, workspace, answer):
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="test",
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission=answer,
    )
    result = TaskScorer(task, workspace=workspace).evaluate(state)
    assert state.submission == answer
    return result.score


@pytest.mark.parametrize(
    "answer",
    [
        "runs/300/result.csv",
        "result.csv",
        "/former/local/workspace/runs/300/result.csv",
        "/results/corral/jobs/remote-job/runs/300/result.csv",
        '{"vdos": "runs/300/result.csv"}',
        '"runs/300/result.csv"',
    ],
)
def test_md_scores_nested_files_in_restored_workspace(answer, tmp_path, monkeypatch):
    workspace = tmp_path / "restored"
    result = workspace / "runs/300/result.csv"
    result.parent.mkdir(parents=True)
    source = next((PACKAGE_DATA_ROOT / "ground_truth/level_2/task_6").glob("*.csv"))
    shutil.copyfile(source, result)
    # A second basename must not override an explicit directory or suffix.
    if answer != "result.csv":
        (workspace / "other").mkdir()
        (workspace / "other/result.csv").write_text("invalid CSV")
    monkeypatch.chdir(tmp_path)
    task = load_tasks_from_json(
        PACKAGE_DATA_ROOT / "level_2/tasks_json/task_6.json", str(tmp_path / "former")
    )["level_2_task_6"]
    assert evaluate(task, workspace, answer) == 1.0


def test_json_values_and_manifest_paths_are_resolved(tmp_path):
    workspace = tmp_path / "restored"
    outputs = workspace / "results"
    outputs.mkdir(parents=True)
    (outputs / "300.traj").touch()
    (outputs / "400.traj").touch()
    manifest = outputs / "manifest.json"
    content = {"300": "300.traj", "nested": ["results/400.traj", 2, "3.14"]}
    manifest.write_text(json.dumps(content))
    original = manifest.read_bytes()
    for answer in ("results/manifest.json", str(manifest)):
        resolved = json.loads(resolve_submission(answer, workspace))
        assert resolved == {
            "300": str(outputs / "300.traj"),
            "nested": [str(outputs / "400.traj"), 2, "3.14"],
        }
    assert manifest.read_bytes() == original


def test_md_scores_snapshot_after_original_workspace_is_deleted(tmp_path):
    original = tmp_path / "original"
    result = original / "nested/result.csv"
    result.parent.mkdir(parents=True)
    source = next((PACKAGE_DATA_ROOT / "ground_truth/level_2/task_6").glob("*.csv"))
    shutil.copyfile(source, result)
    task = load_tasks_from_json(
        PACKAGE_DATA_ROOT / "level_2/tasks_json/task_6.json", str(original)
    )["level_2_task_6"]
    manager = WorkspaceManager(artifact_root=tmp_path / "artifacts")

    async def score_snapshot():
        snapshot = await manager.snapshot(original)
        shutil.rmtree(original)
        async with manager.temporary_materialization(snapshot) as restored:
            return evaluate(task, restored, json.dumps({"vdos": str(result)}))

    assert asyncio.run(score_snapshot()) == 1.0
    assert not original.exists()


@pytest.mark.parametrize("answer", ["143.38", '"143.38"'])
def test_numeric_md_submissions_are_preserved(answer, tmp_path):
    task = next(
        iter(
            load_tasks_from_json(
                PACKAGE_DATA_ROOT / "level_1/tasks_json/task_3.json", str(tmp_path)
            ).values()
        )
    )
    assert evaluate(task, tmp_path, answer) == 1.0


def test_phonon_json_file_and_inline_coefficients(tmp_path):
    task = load_tasks_from_json(
        PACKAGE_DATA_ROOT / "level_2/tasks_json/task_5.json", str(tmp_path)
    )["level_2_task_5"]
    reference = PACKAGE_DATA_ROOT / "ground_truth/level_2/task_5/phonon.json"
    result = tmp_path / "results/phonon.json"
    result.parent.mkdir()
    shutil.copyfile(reference, result)
    for answer in ("results/phonon.json", result.read_text()):
        assert evaluate(task, tmp_path, answer) == 1.0


def test_ambiguous_basename_is_rejected(tmp_path):
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "result.csv").touch()
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_submission("result.csv", tmp_path)


def test_missing_file_does_not_read_from_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "result.csv").write_text("outside workspace")
    monkeypatch.chdir(tmp_path)
    for answer in ("result.csv", str(tmp_path / "result.csv")):
        with pytest.raises(FileNotFoundError, match="missing from the workspace"):
            resolve_submission(answer, workspace)


def test_submission_paths_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.csv"
    secret.touch()
    (workspace / "link.csv").symlink_to(secret)
    for answer in ("../secret.csv", "link.csv", '{"result": "link.csv"}'):
        with pytest.raises(ValueError, match="workspace"):
            resolve_submission(answer, workspace)
