"""Every argument has a line in its docstring (scripts/check_docstring_args.py, #3938)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import check_docstring_args as check  # noqa: E402

GOOD = '''
def scale(x, factor=2.0, *rest, **opts):
    """Scale ``x``.

    Args:
        x: The number.
        factor (float): How much. A long description that runs on
            to a second line, which isn't an entry.
        *rest: More numbers.
        **opts: Options.
    """


class Box:
    """A box.

    Args:
        size: Its size.
    """

    def __init__(self, size):
        self.size = size

    def area(self):
        """No arguments but self, so no Args section."""


def _private():
    pass
'''

BAD = '''
def missing(x, y):
    """Only one.

    Args:
        x: The first.
    """


def extra(x):
    """One too many.

    Args:
        x: The first.
        y: Not an argument.
    """


def none(x):
    """No section at all."""


def outer():
    """Fine."""

    def inner(z):
        return z
'''


def test_a_described_function_passes():
    assert check.offenders("good.py", GOOD) == []


def test_each_gap_is_named_by_its_qualified_name():
    assert check.offenders("bad.py", BAD) == [
        "bad.py::missing",
        "bad.py::extra",
        "bad.py::none",
        "bad.py::outer.inner",
    ]


def test_the_baseline_only_shrinks(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("a.py::old\na.py::fixed\n")
    monkeypatch.setattr(check, "BASELINE", baseline)
    monkeypatch.setattr(sys, "argv", ["check"])
    monkeypatch.setattr(check, "every_offender", lambda: ["a.py::old", "a.py::new"])
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "a.py::new: describe every argument" in out  # a new gap fails
    assert "a.py::fixed: now passes or is gone" in out  # and so does a stale line
    monkeypatch.setattr(check, "every_offender", lambda: ["a.py::old", "a.py::fixed"])
    assert check.main() == 0


def test_the_repo_matches_its_baseline():
    listed = set(check.BASELINE.read_text(encoding="utf-8").split())
    assert set(check.every_offender()) == listed
