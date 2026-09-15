"""Answer extraction and grading.

A grading bug is the worst kind of failure this environment can have: it shows up
as "the agent improved nothing" and costs days of blaming the agent. These are
golden cases, including the ones that caught real bugs during implementation.
"""

from __future__ import annotations

import pytest
from inference_opt.scoring_specs import (
    NO_ANSWER,
    canonical_completion,
    extract_mcq_letters,
    extract_numeric,
    spec_for,
)

CHOICES = ("Paris", "Berlin", "London", "Rome")


class TestMultipleChoice:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("B", ["B"]),
            ("b", ["B"]),
            ("ANSWER: C", ["C"]),
            ("Answer: C.", ["C"]),
            ("[ANSWER]D[/ANSWER]", ["D"]),
            ("Reasoning...\nANSWER: B\n", ["B"]),
            (r"so \boxed{D}", ["D"]),
            ("The final answer is (A)", ["A"]),
            ("I'll pick option C because", ["C"]),
            # A small model often answers with the option text, not its letter.
            ("Paris", ["A"]),
            ("The capital is Berlin", ["B"]),
            ("The answer is Paris", ["A"]),
        ],
    )
    def test_answers_are_found(self, completion, expected):
        assert extract_mcq_letters(completion, n_choices=4, choices=CHOICES) == expected

    @pytest.mark.parametrize("completion", ["no idea", "hmm", "xyz", "", "   "])
    def test_prose_is_not_mistaken_for_a_letter(self, completion):
        # "no idea" contains a valid choice letter; a length-based rule read it as A.
        assert extract_mcq_letters(completion, n_choices=4, choices=CHOICES) == []

    @pytest.mark.parametrize("completion", ["A, C", "ANSWER: A, C"])
    def test_hedged_single_answers_count_as_unanswered(self, completion):
        # Taking the first letter would hand a tied self-consistency vote a coin flip.
        assert extract_mcq_letters(completion, n_choices=4, choices=CHOICES) == []

    @pytest.mark.parametrize(
        ("completion", "expected"),
        [("ANSWER: A, C, D", ["A", "C", "D"]), ("[ANSWER]ABD[/ANSWER]", ["A", "B", "D"]),
         ("A C", ["A", "C"]), ("B", ["B"]), ("nonsense here", [])],
    )
    def test_multi_select(self, completion, expected):
        assert extract_mcq_letters(completion, n_choices=6, multi=True) == expected

    def test_letters_beyond_the_option_count_are_ignored(self):
        assert extract_mcq_letters("ANSWER: Z", n_choices=4, choices=CHOICES) == []


class TestNumeric:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("42", "42"),
            ("ANSWER: 42", "42"),
            ("#### 18", "18"),
            ("The answer is $1,200", "1200"),
            (r"\boxed{3.5}", "3.5"),
            ("He has 3 apples and 2 pears, so 5 total.", "5"),
            ("-7.25", "-7.25"),
            ("2e3", "2e3"),
        ],
    )
    def test_numbers_are_found(self, completion, expected):
        assert extract_numeric(completion) == expected

    def test_no_number_returns_none(self):
        assert extract_numeric("nothing here") is None


class TestCanonicalCompletion:
    def test_mcq_is_rewritten_for_inspect(self):
        assert (
            canonical_completion(
                "I think Berlin", answer_format="mcq_single",
                choices=("Paris", "Berlin"), n_choices=2,
            )
            == "ANSWER: B"
        )

    def test_unparseable_is_explicit(self):
        assert canonical_completion(None, answer_format="numeric").endswith(NO_ANSWER)

    def test_oversized_answers_are_truncated(self):
        # A policy returning megabytes must not bloat the log or the scorer.
        assert len(canonical_completion("x" * 200_000, answer_format="text")) < 9_000

    def test_non_string_answers_are_accepted(self):
        assert canonical_completion(42, answer_format="numeric") == "ANSWER: 42"


class TestSpecSelection:
    def test_chembench_numerics_use_tolerance(self):
        # Upstream chembench grades these on tolerance; exact match would
        # understate every model including the baseline.
        assert spec_for("chembench", "numeric").answer_format == "numeric_tolerant"

    def test_other_numerics_are_exact(self):
        assert spec_for("gsm8k", "numeric").answer_format == "numeric"

    def test_multi_select_is_flagged(self):
        assert spec_for("chembench", "mcq_multi").multiple_correct
