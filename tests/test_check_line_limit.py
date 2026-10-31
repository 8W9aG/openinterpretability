"""Tests for the function-length limit in scripts/check_line_limit.py."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_line_limit", Path(__file__).parents[1] / "scripts" / "check_line_limit.py"
)
check_line_limit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_line_limit)


def _function(name: str, body_lines: int, *, docstring_lines: int = 0, marker: str = "") -> str:
    out = [f"def {name}():{marker}"]
    if docstring_lines:
        out += ['    """'] + ["    doc"] * (docstring_lines - 2) + ['    """']
    out += ["    x = 1"] * body_lines
    return "\n".join(out) + "\n"


def test_length_skips_decorators_and_docstring():
    source = "@click.option('--a')\n@click.option('--b')\n" + _function("f", 3, docstring_lines=5)
    assert check_line_limit.function_lengths(source) == {"f": (4, False)}


def test_wrapped_signature_counts_as_one_line():
    source = "def f(\n    a,\n    b,\n):\n    x = 1\n    return x\n"
    assert check_line_limit.function_lengths(source) == {"f": (3, False)}


def test_methods_and_nested_functions_are_qualified():
    source = "class C:\n    def m(self):\n        def inner():\n            pass\n"
    assert set(check_line_limit.function_lengths(source)) == {"C.m", "C.m.inner"}


def test_marker_exempts_function():
    source = _function("f", 1, marker="  # long-function: reads better whole")
    assert check_line_limit.function_lengths(source)["f"] == (2, True)


def _check(tmp_path, monkeypatch, source: str) -> list[str]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.py").write_text(source)
    return check_line_limit.check_functions(["m.py"])


def test_long_function_fails(tmp_path, monkeypatch):
    (error,) = _check(tmp_path, monkeypatch, _function("f", 40))
    assert error.startswith("m.py::f: 41 lines (max 40)")


def test_function_at_limit_passes(tmp_path, monkeypatch):
    assert _check(tmp_path, monkeypatch, _function("f", 39)) == []


def test_marked_long_function_passes(tmp_path, monkeypatch):
    source = _function("f", 60, marker="  # long-function: a table that reads better whole")
    assert _check(tmp_path, monkeypatch, source) == []
