"""Explicit experiment and student-call budgets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    """Mutable budget shared by one teacher task."""

    max_experiments: int = 20
    max_debug_questions: int = 10
    max_student_calls: int = 250
    experiments: int = 0
    debug_questions: int = 0
    student_calls: int = 0

    def reserve_experiment(self, calls: int) -> None:
        if self.experiments >= self.max_experiments:
            raise RuntimeError("experiment budget exhausted")
        self._reserve_calls(calls)
        self.experiments += 1

    def reserve_student_call(self) -> None:
        """Charge one actual target-model request."""
        self._reserve_calls(1)

    def reserve_debug(self, calls: int = 1) -> None:
        if self.debug_questions >= self.max_debug_questions:
            raise RuntimeError("debug-question budget exhausted")
        self._reserve_calls(calls)
        self.debug_questions += 1

    def _reserve_calls(self, calls: int) -> None:
        if calls < 0 or self.student_calls + calls > self.max_student_calls:
            raise RuntimeError("student inference budget exhausted")
        self.student_calls += calls

    def snapshot(self) -> dict[str, int]:
        return {
            "experiments": self.experiments,
            "debug_questions": self.debug_questions,
            "student_calls": self.student_calls,
            "max_experiments": self.max_experiments,
            "max_debug_questions": self.max_debug_questions,
            "max_student_calls": self.max_student_calls,
        }
