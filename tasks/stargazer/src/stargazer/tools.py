"""Agent tools for iterative Stargazer analysis and model submission."""

from __future__ import annotations

import ast
import base64
import builtins
import importlib
import io
import multiprocessing as mp
import os
import re
import sys
import sysconfig
import tempfile
import threading
import traceback
import types
from contextlib import redirect_stdout, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import scipy  # noqa: F401 - preload numerical extensions before the worker audit hook
from scipy import optimize, signal  # noqa: F401 - safe lazy-import support

from corral.core.tool import Tool

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_OUTPUT_CHARS = 5_000 + len("...(output truncated)")
_MAX_CODE_CHARS = 50_000
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 10.0
_WORKER_START_TIMEOUT_SECONDS = 30.0
_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024**3
_ALLOWED_IMPORTS = {
    "baselines",
    "celerite2",
    "rebound",
    "collections",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "scipy",
    "statistics",
}
_SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "print",
    "AssertionError",
    "Exception",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_BLOCKED_MODULE_ATTRIBUTES = {
    "DataSource",
    "ctypes",
    "ctypeslib",
    "distutils",
    "dump",
    "dumps",
    "f2py",
    "fromfile",
    "genfromtxt",
    "io",
    "load",
    "load_library",
    "loads",
    "loadtxt",
    "memmap",
    "open_memmap",
    "popen",
    "save",
    "savetxt",
    "source",
    "system",
    "tofile",
}


class UnsafeAnalysisCode(ValueError):
    """Raised when REPL code attempts unsupported introspection or I/O."""


class AnalysisTimeoutError(TimeoutError):
    """Raised when an analysis worker exceeds its per-call time budget."""


class _SafeModuleProxy:
    """Expose numerical module attributes while denying I/O escape hatches."""

    __slots__ = ("__wrapped_module",)

    def __init__(self, module: types.ModuleType):
        object.__setattr__(self, "_SafeModuleProxy__wrapped_module", module)

    def __getattribute__(self, name: str) -> Any:
        module = object.__getattribute__(self, "_SafeModuleProxy__wrapped_module")
        if module.__name__ == "json" and name in {"dumps", "loads"}:
            return getattr(module, name)
        if name.startswith("_") or name in _BLOCKED_MODULE_ATTRIBUTES:
            raise AttributeError(f"Module attribute {name!r} is unavailable")
        value = getattr(module, name)
        return _safe_module(value)

    def __repr__(self) -> str:
        module = object.__getattribute__(self, "_SafeModuleProxy__wrapped_module")
        return f"<restricted module {module.__name__!r}>"


_MODULE_PROXIES: dict[str, _SafeModuleProxy] = {}


def _safe_module(value: Any) -> Any:
    if not isinstance(value, types.ModuleType):
        return value
    proxy = _MODULE_PROXIES.get(value.__name__)
    if proxy is None:
        proxy = _SafeModuleProxy(value)
        _MODULE_PROXIES[value.__name__] = proxy
    return proxy


def _restore_module_proxy(name: str) -> _SafeModuleProxy:
    return _safe_module(
        baselines if name == "baselines" else importlib.import_module(name)
    )


def _restore_session_function(
    code: types.CodeType,
    namespace: dict[str, Any],
    safe_builtins: dict[str, Any],
    closure: tuple[types.CellType, ...] | None,
) -> types.FunctionType:
    # FunctionType captures its builtins at construction. Ordinary cloudpickle
    # restores unrestricted builtins and gives functions a separate globals dict.
    namespace["__builtins__"] = safe_builtins
    return types.FunctionType(code, namespace, closure=closure)


def _restore_function_state(function: types.FunctionType, state: tuple) -> None:
    attributes, closure = state
    for name, value in attributes.items():
        setattr(function, name, value)
    if closure is not None:
        for target, source in zip(function.__closure__, closure, strict=True):
            with suppress(ValueError):  # A closure cell can be empty.
                target.cell_contents = source.cell_contents


class _SessionPickler(cloudpickle.CloudPickler):
    """Keep notebook functions attached to the restored, restricted namespace."""

    def __init__(self, file: io.BytesIO, namespace: dict[str, Any]):
        super().__init__(file)
        self.namespace = namespace
        self.closure_cells: dict[int, types.CellType] = {}

    def reducer_override(self, value: Any) -> Any:
        if isinstance(value, _SafeModuleProxy):
            module = object.__getattribute__(value, "_SafeModuleProxy__wrapped_module")
            return _restore_module_proxy, (module.__name__,)
        if (
            isinstance(value, types.FunctionType)
            and value.__globals__.get("__name__") == "__stargazer_session__"
        ):
            attributes = {
                name: getattr(value, name)
                for name in (
                    "__name__",
                    "__qualname__",
                    "__defaults__",
                    "__kwdefaults__",
                    "__annotations__",
                    "__dict__",
                    "__module__",
                    "__doc__",
                )
            }
            # Shared nonlocal variables need shared cells. Empty placeholders
            # allow a closure to refer recursively to its own function.
            closure = (
                tuple(
                    self.closure_cells.setdefault(id(cell), types.CellType())
                    for cell in value.__closure__
                )
                if value.__closure__ is not None
                else None
            )
            return (
                _restore_session_function,
                (
                    value.__code__,
                    value.__globals__,
                    value.__globals__["__builtins__"],
                    closure,
                ),
                (attributes, value.__closure__),
                None,
                None,
                _restore_function_state,
            )
        return super().reducer_override(value)


def _snapshot_namespace(namespace: dict[str, Any]) -> str:
    output = io.BytesIO()
    # Captured print belongs to a single call, including its output buffer.
    captured_print = namespace["__builtins__"].pop("print", None)
    try:
        # Preserve the legacy module RNG as well as Generator objects in locals.
        random_state = np.random.get_state()  # noqa: NPY002
        _SessionPickler(output, namespace).dump((namespace, random_state))
    finally:
        if captured_print is not None:
            namespace["__builtins__"]["print"] = captured_print
    return base64.b64encode(output.getvalue()).decode("ascii")


class AnalysisSession:
    """Handle for a persistent, public-data-only analysis worker process."""

    def __init__(
        self,
        public_data: dict[str, Any],
        *,
        execution_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ):
        if execution_timeout_seconds <= 0.0:
            raise ValueError("execution_timeout_seconds must be positive")
        self._public_data = public_data
        self._execution_timeout_seconds = float(execution_timeout_seconds)
        self._connection: Any = None
        self._process: Any = None
        self._worker_directory: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.RLock()

    def _stop_worker(self) -> None:
        connection, process = self._connection, self._process
        worker_directory = self._worker_directory
        self._connection = None
        self._process = None
        self._worker_directory = None
        if connection is not None:
            with suppress(OSError, ValueError):
                connection.close()
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=1.0)
                process.close()
            except (AssertionError, OSError, ValueError):
                pass
        if worker_directory is not None:
            with suppress(OSError, PermissionError):
                worker_directory.cleanup()

    def _start_worker(self) -> None:
        self._stop_worker()
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        worker_directory = tempfile.TemporaryDirectory(
            prefix="corral-stargazer-analysis-"
        )
        process = context.Process(
            target=_analysis_worker,
            args=(child_connection, self._public_data, worker_directory.name),
            daemon=True,
            name="stargazer-analysis",
        )
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            worker_directory.cleanup()
            raise
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        self._worker_directory = worker_directory
        if not parent_connection.poll(_WORKER_START_TIMEOUT_SECONDS):
            self._stop_worker()
            raise RuntimeError("The Stargazer analysis worker did not start")
        try:
            ready = parent_connection.recv()
        except EOFError as exc:
            self._stop_worker()
            raise RuntimeError("The Stargazer analysis worker failed to start") from exc
        if ready != {"status": "ready"}:
            self._stop_worker()
            raise RuntimeError(f"The Stargazer analysis worker failed: {ready!r}")

    def execute(self, code: str, history: list[dict[str, Any]] | None = None) -> str:
        """Run code in the persistent worker, terminating it on timeout."""
        if len(code) > _MAX_CODE_CHARS:
            raise UnsafeAnalysisCode(
                f"Analysis code is limited to {_MAX_CODE_CHARS:,} characters"
            )
        return str(
            self._request({"command": "execute", "code": code, "history": history})
        )

    def protocol_acknowledged(self) -> bool:
        """Read only the protocol boolean from the worker, never decode its checkpoint."""
        if self._process is None:
            return False
        return bool(self._request({"command": "protocol_ack"}))

    def snapshot(self) -> str | None:
        """Export JSON-compatible state; a stopped or timed-out worker is empty."""
        with self._lock:
            if self._process is None or not self._process.is_alive():
                return None
            return self._request({"command": "snapshot"})

    def restore(self, checkpoint: str | None) -> None:
        """Restore state in an isolated worker, never unpickling in the controller."""
        with self._lock:
            self._stop_worker()
            if checkpoint is not None:
                self._request({"command": "restore", "checkpoint": checkpoint})

    def _request(self, request: dict[str, Any]) -> Any:
        with self._lock:
            process = self._process
            if process is None or not process.is_alive():
                self._start_worker()
            connection = self._connection
            assert connection is not None
            try:
                connection.send(request)
                if not connection.poll(self._execution_timeout_seconds):
                    timeout = self._execution_timeout_seconds
                    self._stop_worker()
                    raise AnalysisTimeoutError(
                        f"Analysis exceeded {timeout:g} seconds; "
                        "the persistent session was reset"
                    )
                response = connection.recv()
            except AnalysisTimeoutError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker()
                raise RuntimeError(
                    "The analysis worker stopped unexpectedly; "
                    "the persistent session was reset"
                ) from exc
            if isinstance(response, dict) and "error" in response:
                self._stop_worker()
                raise RuntimeError(response["error"])
            if not isinstance(response, dict) or "result" not in response:
                self._stop_worker()
                raise RuntimeError(
                    "The analysis worker returned an invalid response; "
                    "the persistent session was reset"
                )
            return response["result"]

    def close(self) -> None:
        """Release the worker process and its private IPC channel."""
        with self._lock:
            connection = self._connection
            process = self._process
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send({"command": "close"})
                    process.join(timeout=1.0)
                except (BrokenPipeError, OSError):
                    pass
            self._stop_worker()

    def __del__(self) -> None:
        with suppress(Exception):
            self._stop_worker()


def _restricted_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    if name == "baselines":
        return _safe_module(baselines)
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORTS:
        if root in PRELOADED_VARS:
            # A preloaded variable is not an importable module. Preserve the
            # original REPL's corrective hint for this common agent mistake.
            raise ModuleNotFoundError(f"No module named '{root}'", name=root)
        raise ImportError(
            f"Import of {root!r} is disabled in the Stargazer analysis session"
        )
    if any(part in _BLOCKED_MODULE_ATTRIBUTES for part in name.split(".")):
        raise ImportError(f"Import of {name!r} is disabled in the analysis session")
    return _safe_module(builtins.__import__(name, globals_, locals_, fromlist, level))


def _safe_builtins() -> dict[str, Any]:
    values = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    values["__import__"] = _restricted_import
    return values


def create_analysis_session(
    *,
    times_days: tuple[float, ...],
    rvs_ms: tuple[float, ...],
    sigmas_ms: tuple[float, ...],
    instruments: tuple[str, ...],
    star_mass_sun: float,
    execution_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
) -> AnalysisSession:
    """Create a persistent worker containing only public trial data."""
    times = tuple(float(value) for value in times_days)
    if not times:
        raise ValueError("At least one observation is required")
    public_data: dict[str, Any] = {
        "times_days": times,
        "rvs_ms": tuple(float(value) for value in rvs_ms),
        "sigmas_ms": tuple(float(value) for value in sigmas_ms),
        "instruments": tuple(str(value) for value in instruments),
        "star_mass_sun": float(star_mass_sun),
    }
    lengths = {
        len(public_data["times_days"]),
        len(public_data["rvs_ms"]),
        len(public_data["sigmas_ms"]),
        len(public_data["instruments"]),
    }
    if len(lengths) != 1:
        raise ValueError("Observation arrays must have equal lengths")
    return AnalysisSession(
        public_data,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def _create_worker_namespace(public_data: dict[str, Any]) -> dict[str, Any]:
    """Materialize a numerical namespace inside the isolated worker."""
    times = np.asarray(public_data["times_days"], dtype=float).copy()
    return {
        "__builtins__": _safe_builtins(),
        "__name__": "__stargazer_session__",
        "np": _safe_module(np),
        "times_days": times,
        "rvs_ms": np.asarray(public_data["rvs_ms"], dtype=float).copy(),
        "sigmas_ms": np.asarray(public_data["sigmas_ms"], dtype=float).copy(),
        "star_mass_sun": float(public_data["star_mass_sun"]),
        "t_ref_days": float(times[0]),
        "baselines": _safe_module(baselines),
        "history": public_data.get("history") or [],
        "stargazer_planet_from_fit": make_planet_from_fit(
            float(public_data["star_mass_sun"])
        ),
        "STARGAZER_SUBMISSION_GUIDE": STARGAZER_SUBMISSION_GUIDE,
        "_protocol_guide_ack": False,
    }


def _runtime_read_roots() -> tuple[str, ...]:
    """Return interpreter/package roots needed for allow-listed lazy imports."""
    paths = sysconfig.get_paths()
    roots = {
        str(Path(path).resolve())
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
        if (path := paths.get(name))
    }
    roots.add(str(Path(__file__).resolve().parents[1]))
    return tuple(sorted(roots))


def _path_is_within(path: str, roots: tuple[str, ...]) -> bool:
    resolved = str(Path(path).resolve())
    for root in roots:
        try:
            if os.path.commonpath((resolved, root)) == root:
                return True
        except ValueError:
            continue
    return False


def _install_worker_audit_hook() -> Callable[[], None]:
    """Deny worker filesystem, network, and process access outside runtime code."""
    runtime_roots = _runtime_read_roots()
    enabled = [True]
    blocked_prefixes = (
        "ctypes.",
        "ftplib.",
        "http.client.",
        "pty.",
        "socket.",
        "subprocess.",
        "urllib.Request",
    )
    blocked_events = {
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.exec",
        "os.fork",
        "os.kill",
        "os.posix_spawn",
        "os.putenv",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.spawn",
        "os.startfile",
        "os.system",
        "os.truncate",
        "os.unsetenv",
        "os.utime",
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if not enabled[0]:
            return
        if event == "open":
            target = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            write_flags = (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            if (
                isinstance(mode, str)
                and any(marker in mode for marker in ("w", "a", "x", "+"))
            ) or (isinstance(flags, int) and flags & write_flags):
                raise PermissionError("Filesystem writes are unavailable")
            if isinstance(target, int):
                return
            try:
                path = os.fsdecode(target)
            except TypeError as exc:
                raise PermissionError("Filesystem access is unavailable") from exc
            if path == os.devnull or _path_is_within(path, runtime_roots):
                return
            raise PermissionError("Filesystem access is unavailable")
        if event in {"os.listdir", "os.scandir"}:
            target = args[0] if args else None
            if isinstance(target, str | bytes | os.PathLike) and _path_is_within(
                os.fsdecode(target), runtime_roots
            ):
                return
            raise PermissionError("Directory access is unavailable")
        if event in blocked_events or event.startswith(blocked_prefixes):
            raise PermissionError(f"Operation {event!r} is unavailable")

    sys.addaudithook(audit)

    def disable() -> None:
        enabled[0] = False

    return disable


def _apply_worker_resource_limits() -> None:
    """Bound pathological allocations where POSIX address-space limits exist."""
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = _WORKER_ADDRESS_SPACE_BYTES
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        if soft == resource.RLIM_INFINITY or soft > limit:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (OSError, ValueError):
        # Windows lacks `resource`; some macOS/container policies reject
        # RLIMIT_AS changes. The process timeout remains independently killable.
        return


def _analysis_worker(
    connection: Any,
    public_data: dict[str, Any],
    working_directory: str,
) -> None:
    """Serve persistent executions in a process that never receives task truth."""

    def disable_audit() -> None:
        pass

    try:
        os.chdir(working_directory)
        os.environ.clear()
        namespace = _create_worker_namespace(public_data)
        _apply_worker_resource_limits()
        disable_audit = _install_worker_audit_hook()
        connection.send({"status": "ready"})
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if not isinstance(request, dict):
                connection.send({"result": "Invalid analysis-worker request."})
                continue
            command = request.get("command")
            if command == "close":
                return
            if command == "protocol_ack":
                connection.send(
                    {"result": bool(namespace.get("_protocol_guide_ack", False))}
                )
                continue
            if command in {"snapshot", "restore"}:
                try:
                    if command == "snapshot":
                        checkpoint = _snapshot_namespace(namespace)
                        connection.send({"result": checkpoint})
                    else:
                        # Checkpoints may contain Python objects, so decoding is
                        # confined to this public-data-only, audited worker.
                        namespace, random_state = cloudpickle.loads(
                            base64.b64decode(request["checkpoint"], validate=True)
                        )
                        np.random.set_state(random_state)  # noqa: NPY002
                        connection.send({"result": None})
                except BaseException:
                    connection.send({"error": traceback.format_exc(limit=8)})
                continue
            if command != "execute" or not isinstance(request.get("code"), str):
                connection.send({"result": "Invalid analysis-worker request."})
                continue
            try:
                if request.get("history") is not None:
                    namespace["history"] = request["history"]
                result = _execute_persistent(request["code"], namespace)
            except BaseException:
                result = traceback.format_exc(limit=8)
            if len(result) > _MAX_OUTPUT_CHARS:
                result = result[:_MAX_OUTPUT_CHARS] + "\n... output truncated"
            connection.send({"result": result})
    except BaseException:
        with suppress(BaseException):
            connection.send({"status": "error", "error": traceback.format_exc(limit=8)})
    finally:
        disable_audit()
        connection.close()


def _validate_repl_tree(tree: ast.AST) -> None:
    """Reject introspection and known file/process entry points before exec."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeAnalysisCode("Dunder names are unavailable")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__")
            or node.attr in _BLOCKED_MODULE_ATTRIBUTES - {"dumps", "loads"}
        ):
            raise UnsafeAnalysisCode(
                f"Dunder or I/O attribute {node.attr!r} is unavailable "
                "in the analysis session"
            )


def _format_execution_error(error: Exception, code: str) -> str:
    result = "Error Traceback:\n"
    lines = code.split("\n")
    for frame in traceback.extract_tb(error.__traceback__):
        if frame.filename == "<string>":
            result += f"  line {frame.lineno}:\n"
            if 0 < frame.lineno <= len(lines):
                result += f"    {lines[frame.lineno - 1].strip()}\n"
    return result + f"{type(error).__name__}: {error}"


def _execute_persistent(code: str, namespace: dict[str, Any]) -> str:
    """Run the original REPL protocol inside the isolated analysis worker."""
    if "matplotlib" in code:
        return "No plotting is allowed. Code was not executed since it contained 'matplotlib'."
    cleaned = wrap_last_line_with_print(sanitize_input(code))
    try:
        tree = ast.parse(cleaned, mode="exec")
    except SyntaxError:
        pass  # Let exec format the original syntax error.
    else:
        _validate_repl_tree(tree)
    # Stargazer executes each call in merged globals/locals. Functions retain
    # that call's globals, including after Corral checkpoint restoration.
    execution_namespace = dict(namespace)
    conflict = detect_shadowing_callable_conflict(cleaned, execution_namespace)
    if conflict:
        return conflict

    class CappedOutput(io.StringIO):
        truncated = False

        def write(self, value: str) -> int:
            remaining = 5000 - self.tell()
            if remaining > 0:
                super().write(value[:remaining])
            self.truncated |= len(value) > remaining
            return len(value)

    output = CappedOutput()
    execution_namespace["__builtins__"]["print"] = builtins.print
    result = None
    try:
        with redirect_stdout(output):
            exec(cleaned, execution_namespace)
    except ModuleNotFoundError as exc:
        if exc.name in PRELOADED_VARS:
            name = exc.name
            result = (
                f"ModuleNotFoundError: No module named '{name}'\n\n"
                f"HINT: `{name}` is a PRE-LOADED VARIABLE, not a module.\n"
                f"Do NOT import it. Just use it directly:\n"
                f"  CORRECT: print({name})\n"
                f"  WRONG:   from {name} import {name}"
            )
        else:
            result = _format_execution_error(exc, cleaned)
    except Exception as exc:
        result = _format_execution_error(exc, cleaned)
    finally:
        namespace.update(execution_namespace)
    if result is None:
        result = output.getvalue()
        if output.truncated:
            result += "...(output truncated)"
    elif len(result) > 5000:
        result = result[:5000] + "...(output truncated)"
    return (
        result
        or "No output. You likely forgot to print the result. Please use `print(...)` to see any output."
    )


class StargazerTool(Tool):
    """Original tool schema dispatched through the trusted Corral environment."""

    def execute(self, **_kwargs: Any) -> Any:
        raise RuntimeError("Stargazer tools must execute through StargazerEnvironment")


# The pinned implementation uses this fallback because no guide file is shipped.
STARGAZER_SUBMISSION_GUIDE = (
    "Stargazer Submission Guide\\n"
    "1) Preferred fields: P_days, m_sin_i_mjup, e, omega_rad, l_rad\\n"
    "2) Reference epoch: t_ref = times_days[0]\\n"
    "3) Convert M0 to l_rad via l_rad = (Omega_rad + omega_rad + M0) mod 2pi\\n"
    "4) Avoid mixing phase aliases; if using l_rad, treat it as canonical\\n"
    "5) Before submit: verify converted action rv_model residual RMS is near sigma\\n"
)
PRELOADED_VARS = {
    "times_days",
    "rvs_ms",
    "sigmas_ms",
    "np",
    "baselines",
    "history",
    "star_mass_sun",
    "t_ref_days",
    "stargazer_planet_from_fit",
    "STARGAZER_SUBMISSION_GUIDE",
}
REQUIRED_OBS_KEYS = ("times_days", "rvs_ms", "sigmas_ms")


def make_planet_from_fit(star_mass_sun):
    def stargazer_planet_from_fit(
        P_days: float,
        K_ms: float,
        e: float = 0.0,
        omega_rad: float = 0.0,
        M0_rad: float = 0.0,
        inc_rad: float = float(np.pi / 2.0),
        Omega_rad: float = 0.0,
        m_sin_i_mjup: float | None = None,
    ) -> dict[str, float]:
        """Convert fitted Keplerian params into canonical Stargazer planet fields."""
        P_days_f = float(P_days)
        K_ms_f = float(max(0.0, K_ms))
        e_f = float(np.clip(e, 0.0, 0.8))
        omega_f = float(omega_rad % (2.0 * np.pi))
        M0_f = float(M0_rad % (2.0 * np.pi))
        inc_f = float(np.clip(inc_rad, 0.0, np.pi))
        Omega_f = float(Omega_rad % (2.0 * np.pi))
        l_rad = float((Omega_f + omega_f + M0_f) % (2.0 * np.pi))

        if m_sin_i_mjup is None:
            P_years = P_days_f / 365.25
            denom = (
                28.4329
                * (star_mass_sun ** (-2.0 / 3.0))
                * (P_years ** (-1.0 / 3.0))
                / np.sqrt(max(1e-12, 1.0 - e_f * e_f))
            )
            msi = float(np.clip(K_ms_f / denom, 1e-3, 30.0))
        else:
            msi = float(np.clip(m_sin_i_mjup, 1e-3, 30.0))

        return {
            "P_days": P_days_f,
            "m_sin_i_mjup": msi,
            "e": e_f,
            "inc_rad": inc_f,
            "Omega_rad": Omega_f,
            "omega_rad": omega_f,
            "l_rad": l_rad,
        }

    return stargazer_planet_from_fit


def _validate_observation(observation: dict[str, Any], fn_name: str) -> None:
    if not isinstance(observation, dict):
        raise TypeError(
            f"{fn_name} expects a dict observation with keys {REQUIRED_OBS_KEYS}, "
            f"got {type(observation).__name__}"
        )
    missing = [k for k in REQUIRED_OBS_KEYS if k not in observation]
    if missing:
        raise KeyError(f"{fn_name} missing required keys: {missing}")


def baseline_null_model(observation: dict[str, Any]) -> dict[str, Any]:
    _validate_observation(observation, "baseline_null_model")
    np.array(
        observation["times_days"], dtype=float
    )  # Validate the original input contract.
    y = np.array(observation["rvs_ms"], dtype=float)
    s_arr = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s_arr)) or np.any(s_arr <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")
    w = 1.0 / (s_arr**2)
    mu = float(np.sum(w * y) / np.sum(w))
    rv_model = np.full_like(y, mu)
    return {"rv_model": rv_model.tolist(), "planets": []}


def baseline_one_sine(observation: dict[str, Any]) -> dict[str, Any]:
    _validate_observation(observation, "baseline_one_sine")
    t = np.array(observation["times_days"], dtype=float)
    y = np.array(observation["rvs_ms"], dtype=float)
    s = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")

    weights = 1.0 / (s**2 + 1e-6)
    y_mean = float(np.average(y, weights=weights))
    y0 = y - y_mean
    freqs = np.linspace(1 / 300.0, 1 / 2.0, 2000)
    best = None
    for f in freqs:
        omega = 2 * np.pi * f
        X = np.vstack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)]).T
        WX = X * weights[:, None]
        beta = np.linalg.pinv(X.T @ WX) @ (X.T @ (weights * y0))
        model = X @ beta + y_mean
        rss = np.sum(((y - model) / s) ** 2)
        if best is None or rss < best[0]:
            best = (rss, f, model, beta)
    rv_model = best[2]
    P_days = 1.0 / best[1]
    beta_best = best[3]
    amp = float(np.hypot(beta_best[0], beta_best[1]))
    omega = 2.0 * np.pi * best[1]
    # Model is y = gamma + A*sin(w t) + B*cos(w t).
    # Convert to y = gamma + K*cos(M), where M = M0 + w*(t-t_ref), e=0.
    # Using sin(wt+phi) form, M0 = phi - pi/2 + w*t_ref.
    phase_sine = float(np.arctan2(beta_best[1], beta_best[0]))
    t_ref = float(t[0])
    M0 = float((phase_sine - np.pi / 2.0 + omega * t_ref) % (2.0 * np.pi))
    rv_offset_ms = float(y_mean + beta_best[2])
    resid = y - rv_model
    rms_ms = float(np.sqrt(np.mean(resid**2)))
    wrms_ms = float(np.sqrt(np.average(resid**2, weights=weights)))
    P_years = P_days / 365.25
    m_sin_i_estimate = amp / (28.4329 * (P_years ** (-1.0 / 3.0)))
    m_sin_i_estimate = float(np.clip(m_sin_i_estimate, 0.001, 10.0))
    guess = {
        "P_days": float(P_days),
        "m_sin_i_mjup": m_sin_i_estimate,
        "e": 0.0,
        "inc_rad": float(np.pi / 2),
        "Omega_rad": 0.0,
        "omega_rad": 0.0,
        "l_rad": M0,
    }
    return {
        "rv_model": rv_model.tolist(),
        "planets": [guess],
        "period_days": float(P_days),
        "semi_amplitude_ms": amp,
        "phase_rad": M0,
        "rv_offset_ms": rv_offset_ms,
        "rms_ms": rms_ms,
        "wrms_ms": wrms_ms,
    }


baselines = types.ModuleType("baselines")
baselines.baseline_null_model = baseline_null_model
baselines.baseline_one_sine = baseline_one_sine


def sanitize_input(query: str) -> str:
    """Sanitize input to the Python REPL."""
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)

    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(query):
        if not in_string:
            if query[i] in "\"'":
                in_string = True
                string_char = query[i]
                result.append(query[i])
            elif i < len(query) - 1 and query[i : i + 2] == "\\n":
                result.append("\n")
                i += 1
            else:
                result.append(query[i])
        else:
            if query[i] == "\\" and i + 1 < len(query):
                result.append(query[i : i + 2])
                i += 1
            elif query[i] == string_char:
                in_string = False
                result.append(query[i])
            else:
                result.append(query[i])
        i += 1

    return "".join(result)


def wrap_last_line_with_print(code: str) -> str:
    """If last line of code is a single word then wrap it in print."""
    lines = code.strip().split("\n")
    last_line = lines[-1].strip()
    if re.match(r"^[^\s,()]+$", last_line):
        lines[-1] = f"print({last_line})"
    return "\n".join(lines)


def detect_shadowing_callable_conflict(code: str, namespace: dict) -> str | None:
    """Detect assigning to a callable name and then calling it in the same snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    callable_names = {k for k, v in namespace.items() if callable(v)}
    assigned_names = set()
    called_names = set()
    defined_functions = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined_functions.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(
            node.target, ast.Name
        ):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
            if (
                node.func.id in {"least_squares", "minimize", "root", "curve_fit"}
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                callback_name = node.args[0].id
                if callback_name in namespace and not callable(
                    namespace[callback_name]
                ):
                    return (
                        f"Invalid optimizer callback: `{callback_name}` is not callable. "
                        f"It may have been overwritten by a variable. Rename that variable "
                        f"(e.g., `{callback_name}_arr`) or redefine `def {callback_name}(...):`."
                    )

    conflict_names = sorted(
        name
        for name in assigned_names & called_names
        if name in callable_names or name in defined_functions
    )
    if not conflict_names:
        invalid_calls = sorted(
            name
            for name in called_names
            if name not in defined_functions
            and name in namespace
            and not callable(namespace[name])
        )
        if not invalid_calls:
            return None
        bad_name = invalid_calls[0]
        return (
            f"Invalid call detected: `{bad_name}` is not callable in the current session. "
            f"It may have been overwritten by a variable. Rename the variable (e.g., `{bad_name}_arr`) "
            f"or redefine `def {bad_name}(...):` before calling it."
        )

    conflict = conflict_names[0]
    return (
        f"Name shadowing detected: `{conflict}` is assigned and called in the same code block. "
        f"Use a different variable name like `{conflict}_arr` or `{conflict}_vec`."
    )


def create_tools() -> dict[str, Tool]:
    """Original two-tool interface, dispatched by the Corral environment."""
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "PythonREPL",
                "description": "A Python REPL. Use this to execute python code. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. You cannot use matplotlib. No plotting is allowed.\nPackages you can import: np, times_days, rvs_ms, sigmas_ms, baselines, history, star_mass_sun, t_ref_days, stargazer_planet_from_fit, STARGAZER_SUBMISSION_GUIDE.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_code": {
                            "type": "string",
                            "description": "A valid python command.",
                        }
                    },
                    "required": ["input_code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_action",
                "description": "Submit a candidate RV model to the environment. Preferred protocol: Stargazer-native planet fields (P_days, m_sin_i_mjup, e, omega_rad, l_rad). Provide up to 7 planets with P_days > 0.5 and eccentricity between 0 and 0.8. This task currently expects submission_mode='params_and_model'. The environment will evaluate your submission and return a reward with detailed metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "planets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "P_days": {
                                        "type": "number",
                                        "description": "Stargazer planet format: orbital period in days (alternative to period_days).",
                                    },
                                    "period_days": {
                                        "type": "number",
                                        "description": "Orbital period in days (must be > 0.5)",
                                    },
                                    "m_sin_i_mjup": {
                                        "type": "number",
                                        "description": "Stargazer planet format: minimum mass in Jupiter masses (optional; if provided, K is approximated assuming M_star=1Msun).",
                                    },
                                    "semi_amplitude_ms": {
                                        "type": "number",
                                        "description": "Semi-amplitude K in m/s (must be >= 0)",
                                    },
                                    "phase_deg": {
                                        "type": "number",
                                        "description": "Phase in degrees (optional, provide either phase_deg or phase_rad)",
                                    },
                                    "phase_rad": {
                                        "type": "number",
                                        "description": "Phase in radians (optional, provide either phase_deg or phase_rad)",
                                    },
                                    "phase": {
                                        "type": "number",
                                        "description": "Phase alias (optional). Interpreted as radians if |phase|<=2π, else degrees if |phase|<=360.",
                                    },
                                    "phase_frac": {
                                        "type": "number",
                                        "description": "Phase as fraction of an orbit in [0,1) (optional). Converted to radians via 2π*phase_frac.",
                                    },
                                    "l_rad": {
                                        "type": "number",
                                        "description": "Stargazer planet format: mean longitude at t_ref=times_days[0] in radians.",
                                    },
                                    "eccentricity": {
                                        "type": "number",
                                        "description": "Eccentricity (between 0 and 0.8)",
                                    },
                                    "e": {
                                        "type": "number",
                                        "description": "Stargazer planet format: eccentricity (alias for eccentricity).",
                                    },
                                    "omega_rad": {
                                        "type": "number",
                                        "description": "Stargazer planet format: argument of periapsis in radians (optional; matching uses l_rad).",
                                    },
                                    "inc_rad": {
                                        "type": "number",
                                        "description": "Inclination in radians for REBOUND forward model (optional; default pi/2).",
                                    },
                                    "Omega_rad": {
                                        "type": "number",
                                        "description": "Longitude of ascending node in radians for REBOUND forward model (optional; default 0).",
                                    },
                                },
                                "description": "Recommended: use Stargazer native fields (P_days, m_sin_i_mjup, e, omega_rad, l_rad). Legacy aliases are accepted.",
                            },
                            "description": "List of planet hypotheses, sorted by confidence (most confident first).",
                        },
                        "rv_offset_ms": {
                            "type": "number",
                            "description": "Constant RV offset in m/s (optional, will be estimated if not provided).",
                        },
                        "noise_jitter_ms": {
                            "type": "number",
                            "description": "Optional white-noise jitter term in m/s.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Brief rationale for this submission (optional but recommended).",
                        },
                    },
                    "required": ["planets"],
                },
            },
        },
    ]
    return {
        entry["function"]["name"]: StargazerTool(
            name=entry["function"]["name"],
            description=entry["function"]["description"],
            params_json_schema=entry["function"]["parameters"],
            trusted=True,
        )
        for entry in definitions
    }
