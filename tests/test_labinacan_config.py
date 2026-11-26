"""The project's config, and the rule that no generic file holds its values (#6629)."""

import dataclasses
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from labinacan import config as cfg

ROOT = Path(__file__).parents[1]
LABINACAN = ROOT / "labinacan"
# Generic scripts that #6629 makes read the config, before #6630 moves them under labinacan/.
GENERIC_SCRIPTS = [
    ROOT / "scripts" / f"{n}.py"
    for n in (
        "claims",
        "issue_state",
        "linear_sync",
        "next_issue",
        "papers",
        "exchange",
        "brand_images",
    )
]
TEXT = {".py", ".html", ".css", ".js", ".md", ".txt", ".toml", ".json", ".svg", ".xml"}
TEXT |= {".mk", ".yml"}  # labinacan.mk and the generic workflow (#6630)
TEXT_NAMES = {".coveragerc"}  # text files with no suffix

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
    import openinterpretability  # noqa: F401  (its import calls use())
    from openinterpretability.core.paths import (
        CLONE_PYPROJECT,
        DEFAULT_PYPROJECT,
        PACKAGED_PYPROJECT,
    )

    cfg.use(DEFAULT_PYPROJECT, PACKAGED_PYPROJECT, CLONE_PYPROJECT)


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


def test_this_project_reads_its_own_values():
    from openinterpretability.core.project import project_links

    links = project_links()
    assert links["site_url"] == "https://open-interpretability.com/"
    assert links["repo_url"] == "https://github.com/8W9aG/openinterpretability"
    assert (links["project_name"], links["package"]) == (
        "Open Interpretability",
        "openinterpretability",
    )
    assert cfg.env("QUIET") == "OPENINTERPRETABILITY_QUIET"
    c = cfg.config()
    assert (c.repo, c.linear_team, c.maintainer) == ("8W9aG/openinterpretability", "OPE", "8W9aG")
    assert c.hf_dataset == "sackfield/open-interpretability-exchange"


def project_values(c: cfg.ProjectConfig) -> set[str]:
    """The strings of ``c`` that no generic file may hold.

    Args:
        c: A project's config.

    Returns:
        The values, as a generic file would spell them.
    """
    repo = urlparse(c.repo_url).path.strip("/")
    return {
        c.name, c.package, c.command, c.env_prefix, urlparse(c.site_url).netloc, repo,
        c.contact_email, cfg.x_handle(c.x_url), c.maintainer, c.hf_dataset,
    }  # fmt: skip


def short_values(c: cfg.ProjectConfig) -> set[str]:
    """The values of ``c`` short enough to be part of a word, matched as whole words.

    Args:
        c: A project's config.

    Returns:
        The values, such as the Linear team key.
    """
    return {c.linear_team}


def hits(paths, values: set[str], words: set[str] = frozenset()) -> dict[str, list[str]]:
    """Each text file in ``paths`` that holds one of the values, with the values it holds.

    Args:
        paths: The files, or a folder to read whole.
        values: The values, matched without case anywhere.
        words: Short values, matched with their case as whole words.

    Returns:
        ``{path: values}`` for each file with a hit, relative to the repository.
    """
    files = sorted(paths.rglob("*")) if isinstance(paths, Path) else sorted(paths)
    out = {}
    for path in files:
        if (path.suffix in TEXT or path.name in TEXT_NAMES) and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="replace")
            found = {v for v in values if v.lower() in text.lower()}
            found |= {w for w in words if re.search(rf"\b{re.escape(w)}\b", text)}
            if found:
                out[path.name if not path.is_relative_to(ROOT) else str(path.relative_to(ROOT))] = (
                    sorted(found)
                )
    return out


def test_the_check_finds_a_value_in_any_case(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text('URL = "https://EXAMPLE.org/"\n')
    (tmp_path / "pkg" / "b.html").write_text("<p>nothing here</p>")
    (tmp_path / "pkg" / "c.ttf").write_text("example.org")  # not a text file
    (tmp_path / "pkg" / "d.py").write_text('TEAM = "APR"  # not APRIL\n')
    (tmp_path / "pkg" / "e.py").write_text("# April showers\n")
    found = hits(tmp_path / "pkg", {"example.org", "A Project"}, {"APR"})
    assert found == {"a.py": ["example.org"], "d.py": ["APR"]}


def test_no_generic_file_holds_a_value_of_this_project():
    values = project_values(cfg.config())
    assert values >= {"open-interpretability.com", "8W9aG/openinterpretability"}  # a set, not a URL
    assert hits(LABINACAN, values, short_values(cfg.config())) == {}
    assert hits(GENERIC_SCRIPTS, values, short_values(cfg.config())) == {}
    flows = ROOT / ".github" / "workflows"
    skills = ROOT / ".claude" / "skills"
    generic = [ROOT / "labinacan.mk", flows / "labinacan-ci.yml", flows / "links.yml"]  # #6630
    split = ("release", "semantic-versioning", "write-paper")  # each has a project.md
    generic += [skills / name / "SKILL.md" for name in split]
    generic += [
        ROOT / name for name in ("ruff.toml", ".coveragerc", ".diff-cover.toml", ".pymarkdown.yml")
    ]
    assert hits(generic, values, short_values(cfg.config())) == {}


def test_the_paper_and_exchange_scripts_take_their_values_from_the_config():
    sys.path.insert(0, str(ROOT / "scripts"))
    # Imported by name. Ruff sorts a script as first-party only where a folder of its name
    # sits at the root. exchange/ is a local, ignored folder, so the sort would differ.
    exchange = importlib.import_module("exchange")
    papers = importlib.import_module("papers")

    c = cfg.config()
    assert {"name": c.name} == papers.GROUP and c.repo_url == papers.REPO
    assert f"{c.site_url}research/" == papers.SITE
    assert c.hf_dataset == exchange.REPO and c.package in exchange.CITING
    assert f"[{c.name}]({c.repo_url})" in exchange.CARD


def test_the_citation_file_agrees_with_the_config():
    # CITATION.cff is this project's own file (GitHub and Zenodo read it), so it keeps its
    # values. They must be the config's.
    import yaml

    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    c = cfg.config()
    assert cff["repository-code"] == c.repo_url and cff["url"] == c.site_url
    assert cff["title"].startswith(f"{c.name}: ")
    assert c.name in cff["message"] and c.package in cff["message"]


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
