"""The coverage thresholds are set once, in .coveragerc and .diff-cover.toml (#3166, #5855)."""

import configparser
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_changed_lines_must_all_be_covered():
    diff_cover = tomllib.loads((ROOT / ".diff-cover.toml").read_text())["tool"]["diff_cover"]
    assert diff_cover["fail_under"] == 100
    coverage = configparser.ConfigParser()
    coverage.read(ROOT / ".coveragerc")
    assert coverage.getint("report", "fail_under") == 100
    assert coverage.getboolean("run", "branch")


def test_the_tables_left_pyproject():
    # A [tool.coverage] or [tool.diff_cover] table left behind would be a second source (#6630).
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    assert not {"coverage", "diff_cover", "ruff", "pymarkdown"} & tool.keys()


def test_make_and_ci_read_the_threshold_from_its_file():
    # A number on the command line would override the file, and drift from it.
    for path in ("labinacan.mk", ".github/workflows/labinacan-ci.yml"):  # #6630
        text = (ROOT / path).read_text().splitlines()
        runs = [line for line in text if "diff-cover coverage.xml" in line]
        assert runs and all("-c .diff-cover.toml" in r and "--fail-under" not in r for r in runs)
        cov = [line for line in text if "--cov-report=xml" in line]
        assert cov and all("--cov-fail-under" not in r for r in cov)
