"""Trusted policy evaluator for one inference-opt run.

Policy code is trusted inside the Docker trial for the first iteration. This
module is the single task-level execution boundary: Corral owns the outer
process/container lifecycle, while the evaluator owns policy setup,
per-question contexts, metering, artifacts, and scoring.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from inference_opt.runner.__main__ import run
from inference_opt.runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator"]


@dataclass(frozen=True, slots=True)
class PolicyEvaluator:
    """Run trusted teacher-written policy code in the current trial process.

    The evaluator deliberately does not create another security boundary.
    Callers must use it inside Corral's per-trial Docker container. Inspect's
    ``time_limit`` remains the policy-run limit; Docker supplies the outer
    resource and lifecycle boundary.
    """

    def run(self, spec: RunSpec) -> RunSummary:
        # Inspect otherwise writes a process-global trace file under the
        # user's application-data directory. In-process evaluations must keep
        # that artifact task-local and must not collide with another run.
        trace_file = Path(spec.out_dir) / "inspect-trace.log"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with _environment("INSPECT_TRACE_FILE", str(trace_file)):
            return run(spec)


@contextmanager
def _environment(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
