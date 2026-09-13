from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from stargazer.tools import AnalysisTimeoutError, create_analysis_session, create_tools


@pytest.fixture
def analysis_session(simple_task):
    session = create_analysis_session(
        **vars(simple_task.observations),
        star_mass_sun=simple_task.star_mass_sun,
    )
    yield session
    session.close()


def test_original_tool_schemas_are_preserved(protocol_reference):
    actual = list(create_tools().values())
    assert [tool.name for tool in actual] == ["PythonREPL", "submit_action"]
    assert [tool.get_openai_tool_format() for tool in actual] == protocol_reference[
        "tools"
    ]
    assert all(not tool.hidden_args for tool in actual)


def test_repl_and_checkpoints_match_original_reference(
    analysis_session, protocol_reference
):
    for step in protocol_reference["repl"]:
        assert analysis_session.execute(step["code"]) == step["output"], step["code"]
        analysis_session.restore(analysis_session.snapshot())
    assert analysis_session.protocol_acknowledged()


def test_helper_clamps_and_converts_inside_repl(analysis_session):
    result = analysis_session.execute(
        "import json\nprint(json.dumps(stargazer_planet_from_fit(17.25, -4, e=2, inc_rad=-2, Omega_rad=8, M0_rad=7)))"
    )
    planet = json.loads(result)
    assert planet["m_sin_i_mjup"] == 0.001
    assert planet["e"] == 0.8
    assert planet["inc_rad"] == 0
    assert planet["Omega_rad"] == pytest.approx(8 % (2 * np.pi))
    assert planet["l_rad"] == pytest.approx(15 % (2 * np.pi))


def test_namespace_matches_upstream_public_surface(analysis_session):
    assert analysis_session.execute("history").strip() == "[]"
    for name in (
        "benchmark_task",
        "StargazerTask",
        "instruments",
        "scipy",
        "optimize",
        "signal",
    ):
        assert "NameError" in analysis_session.execute(name)


def test_analysis_checkpoint_preserves_random_state(analysis_session):
    analysis_session.execute("np.random.seed(123)\nrng = np.random.default_rng(456)")
    checkpoint = analysis_session.snapshot()
    code = "print((np.random.random(3).tolist(), rng.random(3).tolist()))"
    expected = analysis_session.execute(code)
    analysis_session.restore(checkpoint)
    assert analysis_session.execute(code) == expected


def test_analysis_checkpoint_preserves_shared_recursive_closures(analysis_session):
    analysis_session.execute("""def counter():
    count = 0
    def advance():
        nonlocal count
        count += 1
        return count
    def read():
        return count
    return advance, read
advance, read = counter()
def factorial_factory():
    def factorial(n):
        return n * factorial(n - 1) if n else 1
    return factorial
fact_callable = factorial_factory()""")
    analysis_session.restore(analysis_session.snapshot())
    assert (
        analysis_session.execute("print((advance(), read(), fact_callable(5)))").strip()
        == "(1, 1, 120)"
    )


def test_function_builtins_stay_restricted_after_restore(analysis_session):
    analysis_session.execute(
        'def prohibited():\n    import pathlib\ndef read_file():\n    open("secret")'
    )
    analysis_session.restore(analysis_session.snapshot())
    assert "ImportError" in analysis_session.execute("prohibited()")
    assert "NameError" in analysis_session.execute("read_file()")


@pytest.mark.parametrize(
    "payload",
    [
        "stargazer_planet_from_fit.__closure__[0].cell_contents",
        "np.mean.__globals__",
        'np.loadtxt("secret")',
    ],
)
def test_repl_blocks_introspection_and_io(analysis_session, payload):
    assert "UnsafeAnalysisCode" in analysis_session.execute(payload)


def test_repl_blocks_task_bank_reads(analysis_session):
    bank_file = next(
        (Path(__file__).resolve().parents[1] / "data/synthetic").glob("*.json")
    )
    result = analysis_session.execute(
        f"import json\njson.codecs.open({str(bank_file)!r}).read()"
    )
    assert "PermissionError" in result
    assert "Filesystem access is unavailable" in result
    assert "truth_planets" not in result


def test_repl_proxy_cannot_be_unwrapped(analysis_session):
    assert "AttributeError" in analysis_session.execute(
        "np._SafeModuleProxy__wrapped_module"
    )


def test_timeout_resets_worker(simple_task):
    session = create_analysis_session(
        **vars(simple_task.observations),
        star_mass_sun=simple_task.star_mass_sun,
        execution_timeout_seconds=0.2,
    )
    try:
        session.execute("retained = 42")
        with pytest.raises(AnalysisTimeoutError):
            session.execute("while True:\n    retained = 42")
        assert session.snapshot() is None
        assert not session.protocol_acknowledged()
        assert "NameError" in session.execute("retained")
    finally:
        session.close()
