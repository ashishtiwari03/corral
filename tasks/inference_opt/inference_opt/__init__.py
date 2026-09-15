"""Inference-time optimization environment for frozen student LLMs.

A teacher agent writes a Python policy that improves a frozen student model at test
time - prompting, sampling, verification, memory - and is scored on the improvement
over that student's measured zero-shot baseline on a held-out split.
"""

from inference_opt.api import BudgetExhausted
from inference_opt.policy import PolicyError, discover_policy, validate_policy

__all__ = ["BudgetExhausted", "PolicyError", "discover_policy", "validate_policy"]
