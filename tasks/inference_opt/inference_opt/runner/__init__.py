"""The eval host and the controller-side handle onto it.

``SubprocessRunner`` is the only runner a scored path may use: it starts
``python -m inference_opt.runner`` with a scrubbed environment, so policy code
never executes in the process that holds the budget ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from inference_opt.runner.spec import RunSpec, RunSummary

__all__ = ["RunSpec", "RunSummary", "SubprocessRunner", "run_in_process"]


@dataclass(frozen=True, slots=True)
class SubprocessRunner:
    """Runs a policy in a separate interpreter and reads back its artifacts."""

    python: str = sys.executable
    #: Hard wall-clock ceiling, independent of inspect's own per-sample limit.
    timeout_s: int = 3600

    def run(self, spec: RunSpec, *, spec_path: Path | None = None) -> RunSummary:
        out_dir = Path(spec.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_file = spec.write(spec_path or out_dir / "spec.json")

        environment = {
            name: value
            for name, value in os.environ.items()
            # The child must not be able to reach the frozen dataset or its labels.
            if name not in ("CORRAL_INFERENCE_LABELS_PATH", "CORRAL_INFERENCE_DATA_DIR")
        }
        if spec.base_url:
            environment["VLLM_BASE_URL"] = spec.base_url
        environment.setdefault("PYTHONUNBUFFERED", "1")

        try:
            completed = subprocess.run(
                [self.python, "-m", "inference_opt.runner", str(spec_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired:
            summary = RunSummary(
                run_id=spec.run_id,
                ok=False,
                error=f"timeout: the run exceeded {self.timeout_s}s and was killed",
            )
            summary.write(spec.summary_path)
            return summary

        summary_path = Path(spec.summary_path)
        if summary_path.is_file():
            return RunSummary.read(summary_path)

        # The child died before it could report; surface its stderr rather than
        # inventing a score of zero for what may be an infrastructure fault.
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        summary = RunSummary(
            run_id=spec.run_id,
            ok=False,
            error=f"eval host exited {completed.returncode} without a summary:\n{detail}",
        )
        summary.write(spec.summary_path)
        return summary


def run_in_process(spec: RunSpec) -> RunSummary:
    """In-process runner. **Tests only** — it imports policy code here.

    Never reachable from a scored run; ``score.py`` always uses
    :class:`SubprocessRunner`.
    """
    from inference_opt.runner.__main__ import run

    return run(spec)


def parse_stdout_line(text: str) -> dict[str, object]:
    """Read the single JSON status line the eval host prints."""
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}
