from __future__ import annotations

import pytest
from inference_opt.budget import Budget
from inference_opt.env import create_environments
from inference_opt.policy import PolicyError, validate_policy


def test_inference_opt_has_twelve_tasks_per_level(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRAL_INFERENCE_DATA_DIR", str(tmp_path / "data"))
    for level in (1, 2):
        environments = create_environments(level=level, work_dir=str(tmp_path / "work"))
        assert len(environments) == 12
        model_counts = {
            len(environment.current_task.initial_input["models"])
            for environment in environments.values()
        }
        assert model_counts == ({1} if level == 1 else {2})


def test_budget_charges_actual_student_calls():
    budget = Budget(max_student_calls=2)
    budget.reserve_experiment(0)
    budget.reserve_student_call()
    budget.reserve_student_call()
    with pytest.raises(RuntimeError, match="student inference budget"):
        budget.reserve_student_call()


def test_policy_contract(tmp_path):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "policy.py").write_text(
        "def solve(question, model_client, context):\n    return question\n"
    )
    assert validate_policy(policy) == policy.resolve()

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "policy.py").write_text("answer = 1\n")
    with pytest.raises(PolicyError, match="callable solve"):
        validate_policy(invalid)
