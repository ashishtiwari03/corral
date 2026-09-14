"""Public-data-only REPL adapter for Corral's Docker worker boundary."""

from __future__ import annotations

import base64
import json
import os
import traceback
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

from corral.core.tool import tool
from corral.runtime import permissions
from stargazer.tools import (
    _MAX_CODE_CHARS,
    _MAX_OUTPUT_CHARS,
    _apply_worker_resource_limits,
    _create_worker_namespace,
    _execute_persistent,
    _snapshot_namespace,
)

if TYPE_CHECKING:
    import threading


@tool
def _analysis_step(code: str, public_data: str, checkpoint: str | None) -> str:
    """Restore, execute, and snapshot analysis only after OS privilege dropping."""
    if os.geteuid() == 0:
        raise RuntimeError("Docker analysis must run as an unprivileged worker")
    os.environ.clear()
    _apply_worker_resource_limits()
    data = json.loads(public_data)
    if checkpoint is None:
        namespace = _create_worker_namespace(data)
    else:
        # This pickle is untrusted, including on resumed runs. The controller
        # only stores its opaque string; decoding happens in the jailed worker.
        namespace, random_state = cloudpickle.loads(
            base64.b64decode(checkpoint, validate=True)
        )
        np.random.set_state(random_state)  # noqa: NPY002
    if data.get("history") is not None:
        namespace["history"] = data["history"]
    try:
        output = _execute_persistent(code, namespace)
    except BaseException:
        output = traceback.format_exc(limit=8)
    return json.dumps(
        {
            "output": output[:_MAX_OUTPUT_CHARS],
            "checkpoint": _snapshot_namespace(namespace),
            "protocol_ack": bool(namespace.get("_protocol_guide_ack", False)),
        }
    )


def execute_analysis(
    *,
    code: str,
    public_data: dict[str, Any],
    checkpoint: str | None,
    workspace: str,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Dispatch public observations and opaque state, accepting only JSON back."""
    if len(code) > _MAX_CODE_CHARS:
        raise ValueError(f"Analysis code is limited to {_MAX_CODE_CHARS:,} characters")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("Analysis checkpoint must be an opaque string")
    # No REPL deadline: the caller may still cancel through Corral's worker
    # lifecycle, which kills the worker identity and its descendants.
    response = permissions.run_worker(
        "tool",
        (
            _analysis_step,
            {
                "code": code,
                "public_data": json.dumps(public_data),
                "checkpoint": checkpoint,
            },
        ),
        workspace,
        cancel=cancel,
    )["content"]
    result = json.loads(response)
    if (
        not isinstance(result, dict)
        or set(result) != {"output", "checkpoint", "protocol_ack"}
        or not isinstance(result["output"], str)
        or not isinstance(result["checkpoint"], str)
        or not isinstance(result["protocol_ack"], bool)
    ):
        raise RuntimeError("Invalid Docker analysis result")
    return result
