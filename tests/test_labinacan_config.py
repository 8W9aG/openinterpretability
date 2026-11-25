"""The project's config, and the rule that no generic file holds its values (#6629)."""

from pathlib import Path
from urllib.parse import urlparse

import pytest

from labinacan import config as cfg

ROOT = Path(__file__).parents[1]
LABINACAN = ROOT / "labinacan"
TEXT = {".py", ".html", ".css", ".js", ".md", ".txt", ".toml", ".json", ".svg", ".xml"}

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
        contact_email="team@example.org",
    )  # fmt: skip
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
        c.contact_email, cfg.x_handle(c.x_url),
    }  # fmt: skip


def hits(folder: Path, values: set[str]) -> dict[str, list[str]]:
    """Each text file under ``folder`` that holds one of ``values``, with the values it holds.

    Args:
        folder: The folder to read.
        values: The values, matched without case.

    Returns:
        ``{path: values}`` for each file with a hit.
    """
    out = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix in TEXT and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if found := sorted(v for v in values if v.lower() in text):
                out[str(path.relative_to(folder.parent))] = found
    return out


def test_the_check_finds_a_value_in_any_case(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text('URL = "https://EXAMPLE.org/"\n')
    (tmp_path / "pkg" / "b.html").write_text("<p>nothing here</p>")
    (tmp_path / "pkg" / "c.ttf").write_text("example.org")  # not a text file
    assert hits(tmp_path / "pkg", {"example.org", "A Project"}) == {"pkg/a.py": ["example.org"]}


def test_no_generic_file_holds_a_value_of_this_project():
    values = project_values(cfg.config())
    assert "open-interpretability.com" in values and "8W9aG/openinterpretability" in values
    assert hits(LABINACAN, values) == {}
