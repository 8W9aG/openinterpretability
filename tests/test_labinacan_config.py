"""The project's config: ``labinacan/config.py`` reads both tables (#6629)."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labinacan import config as cfg

ROOT = Path(__file__).parents[1]
PYPROJECT = """
[project]
name = "a-project"
description = "A Project: research on something."
maintainers = [{ name = "A Project", email = "Team <team@example.org>" }]

[project.urls]
Homepage = "https://example.org/"
Repository = "https://github.com/someone/a-project"
X = "https://x.com/aproject/"

[tool.labinacan]
name = "A Project"
package = "aproject"
env_prefix = "APROJECT"
linear_team = "APR"
maintainer = "someone"
hf_dataset = "someone/a-project-inputs"

[tool.labinacan.brand]
folder = "brand"
tile = "#000000"
ink = { light = "#111111", dark = "#EEEEEE" }
colours = { red = "#FF0000" }
cells = [[0, 0, 3, "red"], [1, 0, 0, ""]]
"""


@pytest.fixture
def restore():
    """Point the config back at this repository after a test moves it."""
    yield
    cfg.use(ROOT / "pyproject.toml")


def test_the_config_reads_both_tables_of_the_first_path_that_has_them(tmp_path, restore):
    no_table = tmp_path / "other.toml"
    no_table.write_text('[project]\nname = "x"\n')
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT)
    cfg.use(tmp_path / "missing.toml", no_table, path)
    assert cfg.config() == cfg.ProjectConfig(
        name="A Project", package="aproject", command="a-project", env_prefix="APROJECT",
        description="research on something.", site_url="https://example.org/",
        repo_url="https://github.com/someone/a-project", x_url="https://x.com/aproject/",
        contact_email="team@example.org", linear_team="APR", maintainer="someone",
        hf_dataset="someone/a-project-inputs",
        brand=cfg.Brand(folder="brand", tile="#000000",
                        ink={"light": "#111111", "dark": "#EEEEEE"}, colours={"red": "#FF0000"},
                        cells=((0, 0, 1.0, "red"), (1, 0, 0.0, ""))),
    )  # fmt: skip
    assert cfg.config().repo == "someone/a-project"
    assert cfg.env("CHROME") == "APROJECT_CHROME"
    assert cfg.site_url() == "https://example.org/"
    links = cfg.project_links()
    assert links["x_handle"] == "@aproject" and links["project_name"] == "A Project"
    assert links["package"] == "aproject" and links["description"] == "research on something."


def test_a_config_with_no_table_is_an_error(tmp_path, restore):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (tmp_path / "bad.toml").write_text("[project\n")
    cfg.use(tmp_path / "pyproject.toml", tmp_path / "bad.toml")
    with pytest.raises(LookupError, match="no \\[tool.labinacan\\] table"):
        cfg.config()


def test_the_shell_lines_quote_each_value_and_name_this_project():
    c = cfg.config()
    lines = cfg.shell_lines(dataclasses.replace(c, name="A 'Project'"))
    assert lines[0] == "NAME='A '\"'\"'Project'\"'\"''"
    assert f"REPO={c.repo}" in cfg.shell_lines(c) and f"PACKAGE={c.package}" in cfg.shell_lines(c)
    out = subprocess.run(
        [sys.executable, "-m", "labinacan.config"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    assert out.splitlines() == cfg.shell_lines(c)
    shell = subprocess.run(
        ["bash", "-c", 'eval "$OUT"; echo "$REPO $MAINTAINER $SITE_URL"'],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OUT": out},
    ).stdout.split()
    assert shell == [c.repo, c.maintainer, c.site_url]


# runpy warns that the module is already imported, which is harmless here.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_the_module_prints_the_shell_lines_when_run(capsys):
    # In this process, so that coverage sees the main guard (the subprocess above it doesn't).
    import runpy

    runpy.run_module("labinacan.config", run_name="__main__")
    assert capsys.readouterr().out.splitlines() == cfg.shell_lines(cfg.config())
