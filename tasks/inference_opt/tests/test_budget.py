"""Budget accounting, including the failure this design exists to prevent."""

from __future__ import annotations

import pytest
from inference_opt.api import BudgetExhausted
from inference_opt.budget import (
    Budget,
    BudgetLedger,
    BudgetSpec,
    QuestionAllocator,
    RunRecord,
    StateLedger,
)


class TestDurableLedger:
    def test_state_survives_a_fresh_tool_instance(self, tmp_path):
        """The test that would have caught the original design bug.

        A non-trusted Corral tool is cloudpickled into a new subprocess on every
        call and only its return value comes back, so a counter held in a
        `create_tools()` closure is silently discarded. That isolation is only
        active under Docker, so an in-memory budget passes locally and loses
        every charge in the real benchmark. Loading a fresh ledger here
        simulates that process boundary.
        """
        first = BudgetLedger.load(tmp_path, BudgetSpec(max_experiments=3, max_student_calls=100))
        first.reserve(experiments=1, calls=40)

        second = BudgetLedger.load(tmp_path)
        assert second.experiments == 1
        assert second.student_calls == 40

    def test_reservation_is_all_or_nothing(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path, BudgetSpec(max_experiments=5, max_student_calls=50))
        ledger.reserve(experiments=1, calls=10)
        before = (ledger.experiments, ledger.student_calls)
        with pytest.raises(BudgetExhausted):
            ledger.reserve(experiments=1, calls=10_000)
        assert (ledger.experiments, ledger.student_calls) == before

    def test_a_corrupt_ledger_fails_closed(self, tmp_path):
        """A corrupt file must not silently reset the agent's spend to zero."""
        ledger = BudgetLedger.load(tmp_path, BudgetSpec(max_student_calls=100))
        ledger.reserve(calls=10)
        (tmp_path / "state" / "ledger.json").write_text("{not json", encoding="utf-8")
        assert BudgetLedger.load(tmp_path).remaining()["student_calls"] == 0

    def test_best_run_tracks_the_largest_delta(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path)
        ledger.record_run(RunRecord(run_id="r1", kind="experiment", policy_dir="policy", delta=0.10))
        ledger.record_run(RunRecord(run_id="r2", kind="experiment", policy_dir="policy", delta=0.25))
        ledger.record_run(RunRecord(run_id="r3", kind="experiment", policy_dir="policy", delta=0.05))
        assert BudgetLedger.load(tmp_path).best_run_id == "r2"

    def test_dry_runs_never_become_the_submission(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path)
        ledger.record_run(RunRecord(run_id="d1", kind="dry_run", policy_dir="policy", delta=0.99))
        assert ledger.best_run_id is None

    def test_refund_returns_unspent_calls(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path, BudgetSpec(max_student_calls=100))
        ledger.reserve(calls=50)
        ledger.refund(calls=30)
        assert BudgetLedger.load(tmp_path).student_calls == 20

    def test_reveals_are_tracked_without_duplicates(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path)
        ledger.mark_revealed(["a", "b"])
        ledger.mark_revealed(["b", "c"])
        assert BudgetLedger.load(tmp_path).revealed_ids == ["a", "b", "c"]

    def test_nag_threshold_reflects_the_tightest_budget(self, tmp_path):
        ledger = BudgetLedger.load(tmp_path, BudgetSpec(max_experiments=10, max_student_calls=100))
        assert ledger.fraction_left() == 1.0
        ledger.reserve(experiments=9)
        assert ledger.fraction_left() == pytest.approx(0.1)


class TestAllocator:
    def test_reserves_protect_later_questions(self, tmp_path):
        allocator = QuestionAllocator(total_calls=100, questions=10, per_question_cap=8)
        allocator.charge("q1", 8)
        assert allocator.allowance("q9") > 0

    def test_per_question_cap_is_hard(self):
        allocator = QuestionAllocator(total_calls=1000, questions=10, per_question_cap=4)
        allocator.charge("q1", 4)
        with pytest.raises(BudgetExhausted, match="per-question cap"):
            allocator.charge("q1", 1)

    def test_shared_pool_runs_out(self):
        allocator = QuestionAllocator(total_calls=10, questions=10, per_question_cap=10)
        for index in range(10):
            allocator.charge(f"q{index}", 1)
        with pytest.raises(BudgetExhausted, match="run call budget exhausted"):
            allocator.charge("q0", 9)

    def test_used_total_is_tracked(self):
        allocator = QuestionAllocator(total_calls=100, questions=5, per_question_cap=8)
        allocator.charge("q1", 3)
        allocator.charge("q2", 2)
        assert allocator.used_total == 5


def test_state_ledger_uses_the_corral_namespace():
    state = {}
    ledger = StateLedger(state, BudgetSpec(max_experiments=2, max_student_calls=10))
    ledger.reserve(experiments=1, calls=3)
    ledger.mark_revealed(["q1"])
    assert state["experiments"] == 1
    assert state["student_calls"] == 3
    assert state["revealed_ids"] == ["q1"]


class TestLegacyBudget:
    def test_scaffold_contract_still_holds(self):
        budget = Budget(max_student_calls=2)
        budget.reserve_experiment(0)
        budget.reserve_student_call()
        budget.reserve_student_call()
        # BudgetExhausted derives from RuntimeError so defensive policies and the
        # original tests both keep working.
        with pytest.raises(RuntimeError, match="student inference budget"):
            budget.reserve_student_call()

    def test_output_token_ceiling(self):
        budget = Budget(max_student_calls=10, max_output_tokens=100)
        budget.record_tokens(60)
        with pytest.raises(BudgetExhausted, match="output-token"):
            budget.record_tokens(50)
