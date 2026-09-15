"""The AST validator is the load-bearing defence; every rule gets a test."""

from __future__ import annotations

import pytest
from inference_opt.validator import ALLOWED_IMPORTS, validate_source, validate_tree

TEMPLATE = __import__("pathlib").Path(
    __import__("inference_opt").__file__
).parent / "templates" / "policy"


def codes(source: str) -> set[str]:
    return {finding.code for finding in validate_source(source).errors}


class TestForbiddenAccess:
    def test_reading_the_target_through_inspect_is_rejected(self):
        # inspect exposes the live TaskState (and therefore the answer) through a
        # contextvar. This import is the only thing standing between a policy and
        # a free 100%.
        source = (
            "from inspect_ai.solver._task_state import sample_state\n"
            "def solve(q, c):\n    return sample_state().target.text\n"
        )
        assert "IO001" in codes(source)

    @pytest.mark.parametrize(
        "module", ["requests", "httpx", "openai", "anthropic", "os", "subprocess",
                   "socket", "urllib", "pathlib", "importlib", "ctypes", "pickle"]
    )
    def test_dangerous_imports_are_rejected(self, module):
        assert "IO001" in codes(f"import {module}\n")

    @pytest.mark.parametrize(
        "call",
        ["eval('1')", "exec('x=1')", "__import__('os')", "open('/etc/passwd')",
         "getattr(o, 'x')", "setattr(o, 'x', 1)", "globals()", "vars(o)",
         "compile('1', '<s>', 'eval')"],
    )
    def test_reflection_and_io_builtins_are_rejected(self, call):
        assert "IO002" in codes(f"def f(o):\n    return {call}\n")

    def test_aliasing_a_forbidden_builtin_is_rejected(self):
        assert "IO002" in codes("sneaky = eval\n")

    def test_dunder_attribute_access_is_rejected(self):
        assert "IO003" in codes("def f(o): return o.__class__.__bases__\n")

    def test_private_attribute_of_another_object_is_rejected(self):
        # The budget meter keeps its counter closure-private; this closes the
        # remaining route to it.
        assert "IO004" in codes("def f(ctx): return ctx.student._meter\n")

    def test_own_private_attributes_are_allowed(self):
        source = (
            "class Policy:\n"
            "    def __init__(self): self._cache = {}\n"
            "    def solve(self, q, ctx): return self._cache.get(q.id, 'A')\n"
        )
        assert validate_source(source).ok

    def test_syntax_error_is_reported_with_a_line(self):
        report = validate_source("def solve(:\n")
        assert not report.ok
        assert report.errors[0].code == "SYN001"


class TestAllowed:
    @pytest.mark.parametrize("module", sorted(ALLOWED_IMPORTS))
    def test_every_allowlisted_module_passes(self, module):
        assert validate_source(f"import {module}\n").ok

    def test_a_realistic_policy_passes_clean(self):
        source = (
            "import re, json, random\n"
            "from collections import Counter\n"
            "from dataclasses import dataclass\n"
            "MANIFEST = {'name': 'vote', 'memory': 'shared'}\n"
            "class Policy:\n"
            "    def setup(self, ctx):\n"
            "        ctx.memory.set('demos', [e.answer for e in ctx.train_examples])\n"
            "    def solve(self, q, ctx):\n"
            "        ctx.scratch['fallback'] = 'A'\n"
            "        outs = ctx.student.sample(q.text, n=3, temperature=0.7)\n"
            "        return Counter(outs).most_common(1)[0][0]\n"
        )
        assert validate_source(source).ok

    def test_local_helper_modules_are_importable(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "policy.py").write_text("from helpers import go\n", encoding="utf-8")
        (root / "helpers.py").write_text("def go(): return 1\n", encoding="utf-8")
        assert validate_tree(root).ok

    def test_relative_imports_are_allowed(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "policy.py").write_text("from . import helpers\n", encoding="utf-8")
        (root / "helpers.py").write_text("x = 1\n", encoding="utf-8")
        assert validate_tree(root).ok


class TestTree:
    def test_shipped_starter_policy_is_valid(self):
        """Whatever the agent starts from must itself obey the rules."""
        report = validate_tree(TEMPLATE)
        assert report.ok, report.render()

    def test_missing_policy_file(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        report = validate_tree(root)
        assert not report.ok
        assert report.errors[0].code == "PKG002"

    def test_violation_in_a_helper_file_is_caught(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "policy.py").write_text("from helpers import go\n", encoding="utf-8")
        (root / "helpers.py").write_text("import requests\n", encoding="utf-8")
        report = validate_tree(root)
        assert not report.ok
        assert report.errors[0].path == "helpers.py"

    def test_report_names_the_file_and_line(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "policy.py").write_text(
            "def solve(q, c):\n    pass\nimport socket\n", encoding="utf-8"
        )
        finding = validate_tree(root).errors[0]
        assert finding.path == "policy.py"
        assert finding.line == 3
        assert "student client" in finding.message
