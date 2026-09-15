"""Static validation of a teacher-written policy tree.

This is the load-bearing defence of the environment. The one rule it enforces is:
*the submitted policy may use only the student client it is handed* — no other
model, no network, no filesystem outside its own artifacts, and no reflection that
would let it reach any of those indirectly.

It is a static check and therefore not a sandbox. It is paired with a scrubbed
child environment, a call-count cross-check against the eval log, and an empirical
probe for answer memorisation. See the environment README for the threat model and
the residual risk.

Run at three points, deliberately: ``dry_run_policy`` (advisory, so the agent learns
the rules while it still has budget), ``submit_policy`` (refuses to stage), and
scoring (refuses to run).
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "ALLOWED_IMPORTS",
    "FORBIDDEN_NAMES",
    "Finding",
    "ValidationReport",
    "validate_source",
    "validate_tree",
]

#: Standard-library modules a policy may import. Deliberately small: everything a
#: prompting strategy legitimately needs (text munging, counting, sampling, maths)
#: and nothing that can open a socket, a file, or another process.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "abc",
        "collections",
        "copy",
        "dataclasses",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "unicodedata",
    }
)

#: Builtins that would defeat the import allowlist or reach the filesystem.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)

#: Names that are safe to read a private attribute from — a policy's own state.
_SELF_NAMES: frozenset[str] = frozenset({"self", "cls"})

_ADVICE = (
    "The student client passed to your policy is the only permitted model access."
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule violation, located precisely enough to fix without guessing."""

    path: str
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The result of validating a policy tree."""

    errors: tuple[Finding, ...] = ()
    warnings: tuple[Finding, ...] = ()
    files_checked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self, limit: int = 20) -> str:
        """A compact report suitable for a tool result."""
        if self.ok and not self.warnings:
            return f"OK - {len(self.files_checked)} file(s) checked, no findings."
        lines: list[str] = []
        if self.errors:
            lines.append(f"{len(self.errors)} error(s):")
            lines += [f"  {finding}" for finding in self.errors[:limit]]
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            lines += [f"  {finding}" for finding in self.warnings[:limit]]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "files_checked": list(self.files_checked),
            "errors": [asdict(finding) for finding in self.errors],
            "warnings": [asdict(finding) for finding in self.warnings],
        }


def _local_module_names(root: Path) -> frozenset[str]:
    """Modules the policy ships itself, which it may of course import.

    The policy directory goes on ``sys.path`` so multi-file policies work, which
    means ``import helpers`` is a local import, not an escape.
    """
    names: set[str] = set()
    for path in root.rglob("*.py"):
        names.add(path.stem)
        if path.name == "__init__.py":
            names.add(path.parent.name)
    for path in root.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            names.add(path.name)
    return frozenset(names)


class _Checker(ast.NodeVisitor):
    def __init__(self, path: str, local_modules: frozenset[str]) -> None:
        self.path = path
        self.local_modules = local_modules
        self.errors: list[Finding] = []
        self.warnings: list[Finding] = []

    def _error(self, node: ast.AST, code: str, message: str) -> None:
        self.errors.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                code=code,
                message=message,
            )
        )

    def _check_module(self, node: ast.AST, module: str) -> None:
        root = module.split(".", maxsplit=1)[0]
        if root in self.local_modules or root in ALLOWED_IMPORTS:
            return
        self._error(
            node,
            "IO001",
            f"import of {module!r} is not permitted. {_ADVICE} "
            f"Permitted modules: {', '.join(sorted(ALLOWED_IMPORTS))}.",
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # `from . import x` / `from .helpers import y` are local by construction.
        if node.level == 0:
            self._check_module(node, node.module or "")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
            self._error(
                node,
                "IO002",
                f"call to {func.id}() is not permitted; it would bypass the "
                f"import rules. {_ADVICE}",
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr.endswith("__"):
            self._error(
                node,
                "IO003",
                f"access to dunder attribute {attr!r} is not permitted; "
                "reflection can reach modules the import rules exclude.",
            )
        elif attr.startswith("_"):
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in _SELF_NAMES):
                self._error(
                    node,
                    "IO004",
                    f"access to private attribute {attr!r} on another object is "
                    "not permitted; budget metering relies on it.",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Catch a forbidden builtin being aliased rather than called directly,
        # e.g. `f = eval` or `list(map(__import__, names))`.
        if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
            self._error(
                node,
                "IO002",
                f"reference to {node.id} is not permitted. {_ADVICE}",
            )
        self.generic_visit(node)


def validate_source(
    source: str,
    *,
    path: str = "policy.py",
    local_modules: frozenset[str] = frozenset(),
) -> ValidationReport:
    """Validate one module's source text."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return ValidationReport(
            errors=(
                Finding(
                    path=path,
                    line=exc.lineno or 0,
                    column=exc.offset or 0,
                    code="SYN001",
                    message=f"could not parse: {exc.msg}",
                ),
            ),
            files_checked=(path,),
        )
    checker = _Checker(path, local_modules)
    checker.visit(tree)
    return ValidationReport(
        errors=tuple(checker.errors),
        warnings=tuple(checker.warnings),
        files_checked=(path,),
    )


def validate_tree(root: Path | str) -> ValidationReport:
    """Validate every ``*.py`` under a policy directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return ValidationReport(
            errors=(
                Finding(
                    path=str(root),
                    line=0,
                    column=0,
                    code="PKG001",
                    message="policy directory does not exist",
                ),
            )
        )
    entry = root / "policy.py"
    if not entry.is_file():
        return ValidationReport(
            errors=(
                Finding(
                    path=str(entry),
                    line=0,
                    column=0,
                    code="PKG002",
                    message="a policy directory must contain policy.py",
                ),
            )
        )

    local_modules = _local_module_names(root)
    errors: list[Finding] = []
    warnings: list[Finding] = []
    checked: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(
                Finding(
                    path=relative,
                    line=0,
                    column=0,
                    code="PKG003",
                    message=f"could not read: {exc}",
                )
            )
            continue
        report = validate_source(source, path=relative, local_modules=local_modules)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
        checked.append(relative)

    return ValidationReport(
        errors=tuple(errors), warnings=tuple(warnings), files_checked=tuple(checked)
    )
