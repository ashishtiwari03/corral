"""Inference-opt policy evaluation.

The first iteration treats policy code as trusted within the Docker trial and
uses one in-process evaluator.
"""

from __future__ import annotations

from inference_opt.runner.evaluator import PolicyEvaluator
from inference_opt.runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator", "RunSpec", "RunSummary", "run_in_process"]


def run_in_process(spec: RunSpec) -> RunSummary:
    """Compatibility helper for tests and local development."""
    return PolicyEvaluator().run(spec)
