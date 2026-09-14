"""Validation and execution contract for teacher-produced policies."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


class PolicyError(ValueError):
    """Raised when a submitted policy is invalid."""


def _load_policy(path: Path) -> ModuleType:
    policy_file = path / "policy.py"
    if not policy_file.is_file():
        raise PolicyError(f"policy artifact must contain {policy_file.name}")
    spec = importlib.util.spec_from_file_location("submitted_policy", policy_file)
    if spec is None or spec.loader is None:
        raise PolicyError("could not load policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy(path: str | Path) -> Path:
    """Validate an artifact and return its resolved directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise PolicyError(f"policy directory does not exist: {root}")
    module = _load_policy(root)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise PolicyError("policy.py must define callable solve(question, model_client, context)")
    return root


def load_policy(path: str | Path):
    """Load and return the validated policy solve function."""
    root = validate_policy(path)
    return _load_policy(root).solve
