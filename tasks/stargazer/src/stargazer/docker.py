"""Public-data-only REPL adapter for Corral's Docker worker boundary."""

from __future__ import annotations

import base64
import json
import os
import threading
import traceback
from typing import Any

import cloudpickle
import numpy as np

from corral.core.tool import tool
from corral.runtime import permissions
from stargazer.tools import (
    _MAX_CODE_CHARS,
    _MAX_OUTPUT_CHARS,
    _WORKER_START_TIMEOUT_SECONDS,
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    _apply_worker_resource_limits,
    _create_worker_namespace,
    _execute_persistent,
    _install_worker_audit_hook,
    _snapshot_namespace,
)


@tool
def _analysis_step(code: str, public_data: str, checkpoint: str | None) -> str:
    """Restore, execute, and snapshot analysis only after OS privilege dropping."""
    if os.geteuid() == 0:
        raise RuntimeError("Docker analysis must run as an unprivileged worker")
    os.environ.clear()
    _apply_worker_resource_limits()
    disable_audit = _install_worker_audit_hook()
    try:
        if checkpoint is None:
            namespace = _create_worker_namespace(json.loads(public_data))
        else:
            # This pickle is untrusted, including on resumed runs. The controller
            # only stores its opaque string; decoding happens in the jailed worker.
            namespace, random_state = cloudpickle.loads(
                base64.b64decode(checkpoint, validate=True)
            )
            np.random.set_state(random_state)  # noqa: NPY002
        if json.loads(public_data).get("history") is not None:
            namespace["history"] = json.loads(public_data)["history"]
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
    finally:
        disable_audit()


def execute_analysis(
    *, code: str, public_data: dict[str, Any], checkpoint: str | None, workspace: str
) -> dict[str, Any]:
    """Dispatch public observations and opaque state, accepting only JSON back."""
    if len(code) > _MAX_CODE_CHARS:
        raise ValueError(f"Analysis code is limited to {_MAX_CODE_CHARS:,} characters")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("Analysis checkpoint must be an opaque string")
    cancelled = threading.Event()
    # Include fresh worker startup in the wall-clock limit. Cancellation kills
    # the worker identity and its descendants even if Python code never returns.
    timeout = _WORKER_START_TIMEOUT_SECONDS + DEFAULT_ANALYSIS_TIMEOUT_SECONDS
    timer = threading.Timer(timeout, cancelled.set)
    timer.daemon = True
    timer.start()
    try:
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
            cancel=cancelled,
        )["content"]
        result = json.loads(response)
    except RuntimeError:
        if not cancelled.is_set():
            raise
        return {
            "output": f"AnalysisTimeoutError: analysis worker exceeded {timeout:g} seconds including startup; the persistent session was reset",
            "checkpoint": None,
            "protocol_ack": False,
        }
    finally:
        timer.cancel()
    if (
        not isinstance(result, dict)
        or set(result) != {"output", "checkpoint", "protocol_ack"}
        or not isinstance(result["output"], str)
        or not isinstance(result["checkpoint"], str)
        or not isinstance(result["protocol_ack"], bool)
    ):
        raise RuntimeError("Invalid Docker analysis result")
    return result
