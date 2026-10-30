"""Tests for the function-length ratchet in scripts/check_line_limit.py."""

import importlib.util
from pathlib import Path

import pytest

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


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowlist = tmp_path / "long_functions.txt"
    monkeypatch.setattr(check_line_limit, "ALLOWLIST", allowlist)
    return tmp_path, allowlist


def _check(repo, source: str, allowlist: str) -> list[str]:
    root, path = repo
    (root / "m.py").write_text(source)
    path.write_text(allowlist)
    return check_line_limit.check_functions(["m.py"])


def test_new_long_function_fails(repo):
    assert "max 40" in _check(repo, _function("f", 40), "")[0]


def test_function_at_limit_passes(repo):
    assert _check(repo, _function("f", 39), "") == []


def test_listed_function_may_not_grow(repo):
    assert "grew from 50" in _check(repo, _function("f", 60), "m.py::f 50\n")[0]


def test_listed_function_at_its_length_passes(repo):
    assert _check(repo, _function("f", 49), "# comment\nm.py::f 50\n") == []


def test_stale_entries_fail(repo):
    (shrank,) = _check(repo, _function("f", 44), "m.py::f 50\n")
    assert "lower its entry" in shrank
    (split,) = _check(repo, _function("f", 10), "m.py::f 50\n")
    assert "remove its entry" in split
    (gone,) = _check(repo, _function("g", 10), "m.py::f 50\n")
    assert "no longer exists" in gone
