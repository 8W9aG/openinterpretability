"""Comments that start with a step number fail the lint (scripts/check_step_comments.py)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_step_comments.py"
spec = importlib.util.spec_from_file_location("check_step_comments", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize(
    ("path", "line"),
    [
        ("a.py", "    # 1. Scan evidence."),
        ("a.sh", "# 2) Build."),
        ("ci.yml", "      # 10. Upload."),
        ("Makefile", "# 3. Lint."),
        ("site.js", "  // 1. Read the tokens."),
        ("page.html", "{# 2. The header. #}"),
    ],
)
def test_a_numbered_step_comment_fails(path, line):
    assert check.offences(path, f"x = 1\n{line}\n") == [f"{path}:2: {line.strip()}"]


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("a.py", '"""Steps:\n\n1. Scan.\n2. Review.\n"""'),  # a docstring list
        ("a.py", "x = 1  # 1. inline, after code"),
        ("a.py", "# See §3 step 9."),
        ("a.py", "# 2026 is the year."),  # a number with no step mark
        ("notes.md", "# 1. A Markdown heading"),
        ("a.js", "# 1. not a JavaScript comment"),
    ],
)
def test_other_numbers_pass(path, text):
    assert check.offences(path, text) == []
