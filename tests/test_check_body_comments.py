"""Long function bodies need a comment (scripts/check_body_comments.py, #3159)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_body_comments.py"
spec = importlib.util.spec_from_file_location("check_body_comments", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def _function(name: str, lines: int, comment: bool = False, indent: str = "") -> str:
    body = [f"{indent}    x{i} = {i}" for i in range(lines)]
    if comment:
        body[3] += "  # the reason for this step"
    return f'{indent}def {name}():\n{indent}    """Doc."""\n' + "\n".join(body) + "\n"


def test_a_long_body_with_no_comment_fails_and_a_comment_passes():
    source = (_function("bare", check.MIN_LINES) + _function("explained", check.MIN_LINES, True)
              + _function("short", check.MIN_LINES - 1))  # fmt: skip
    assert check.offenders("m.py", source) == ["m.py::bare"]


def test_methods_are_named_with_their_class():
    source = "class Thing:\n" + _function("run", check.MIN_LINES, indent="    ")
    assert check.offenders("m.py", source) == ["m.py::Thing.run"]


def test_a_stale_baseline_line_fails(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("gone.py::old\n")
    monkeypatch.setattr(check, "BASELINE", baseline)
    monkeypatch.setattr(check, "every_offender", lambda: [])
    monkeypatch.setattr(check.sys, "argv", ["check"])
    assert check.main() == 1
    assert "Delete its line" in capsys.readouterr().out


def test_the_repository_matches_its_baseline():
    listed = set(check.BASELINE.read_text().split())
    assert set(check.every_offender()) == listed
