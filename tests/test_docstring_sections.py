"""Public docstrings document returns, yields and raises (scripts/check_docstring_sections.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import check_docstring_sections as check  # noqa: E402

GOOD = '''
import click


def total(x):
    """Add one.

    Args:
        x: The number.

    Returns:
        ``x`` plus one.

    Raises:
        ValueError: When ``x`` is negative.
    """
    if x < 0:
        raise ValueError(x)
    return x + 1


def items():
    """Each item.

    Yields:
        The numbers 0 and 1.
    """
    yield from range(2)


def nothing():
    """Return nothing, and re-raise what it catches."""
    try:
        pass
    except OSError:
        raise
    return None


class Box:
    @property
    def size(self):
        """The box's size."""
        return 1


def cli():
    """Fail on the command line.

    Raises:
        click.ClickException: Always.
    """
    raise click.ClickException("no")


def _private():
    return 1
'''

BAD = '''
def adds(x):
    """No Returns."""
    return x + 1


def gives():
    """No Yields."""
    yield 1


def fails():
    """Raises the wrong one.

    Raises:
        KeyError: Never.
    """
    raise ValueError("x")


def nested():
    """A nested function's return is its own."""

    def inner():
        return 1

    return None
'''


def test_documented_functions_pass():
    assert check.offenders("good.py", GOOD) == []


def test_each_missing_section_is_named():
    assert check.offenders("bad.py", BAD) == [
        "bad.py::adds: Returns",
        "bad.py::gives: Yields",
        "bad.py::fails: Raises",
    ]


def test_the_baseline_only_shrinks(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("a.py::old\na.py::fixed\n")
    monkeypatch.setattr(check, "BASELINE", baseline)
    monkeypatch.setattr(sys, "argv", ["check"])
    found = ["a.py::old: Returns", "a.py::new: Yields"]
    monkeypatch.setattr(check, "every_offender", lambda: found)
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "a.py::new: Yields: document it" in out and "a.py::fixed: now passes" in out


def test_the_repo_matches_its_baseline():
    listed = set(check.BASELINE.read_text(encoding="utf-8").split())
    assert {line.split(": ")[0] for line in check.every_offender()} == listed
