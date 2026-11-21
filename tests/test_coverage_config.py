"""The coverage thresholds are set once, in pyproject.toml (#3166, #5855)."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_changed_lines_must_all_be_covered():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    assert config["diff_cover"]["fail_under"] == 100
    assert 90 <= config["coverage"]["report"]["fail_under"] <= 100


def test_make_and_ci_read_the_threshold_from_pyproject():
    # A number on the command line would override pyproject.toml, and drift from it.
    for path in ("Makefile", ".github/workflows/ci.yml"):
        text = (ROOT / path).read_text().splitlines()
        runs = [line for line in text if "diff-cover coverage.xml" in line]
        assert runs and all("-c pyproject.toml" in r and "--fail-under" not in r for r in runs)
