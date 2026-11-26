"""labinacan's own site build, for a project that has none (#6631)."""

import runpy
import sys

import pytest

from labinacan import config as cfg
from labinacan.site import build as site_build
from labinacan.site.css import accent_colours
from tests.test_labinacan_config import PYPROJECT, restore  # noqa: F401  (a fixture)

ACCENT = '\naccent = { 400 = "#34d399", 700 = "#047857" }\n'


@pytest.fixture
def project(tmp_path, restore):  # noqa: F811  (the fixture above)
    """A project with no brand, an About page of its own, and one post."""
    no_brand = PYPROJECT[: PYPROJECT.index("[tool.labinacan.brand]")]
    (tmp_path / "pyproject.toml").write_text(no_brand)
    cfg.use(tmp_path / "pyproject.toml")
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "about.html").write_text(
        '{% extends "base.html" %}{% block content %}<p>About {{ project_name }}.</p>{% endblock %}'
    )
    (templates / "_part.html").write_text("a part of another page, not a page")
    return tmp_path


def test_a_project_with_no_templates_of_its_own_builds(project):
    out = project / "dist"
    report = site_build.build(out, templates=None)
    home = (out / "index.html").read_text()
    assert "A Project" in home and "Research on something." in home
    assert (out / "research" / "index.html").exists() and (out / "posts" / "index.html").exists()
    assert (out / "sitemap.xml").exists() and (out / "robots.txt").exists()
    assert not (out / "assets" / "mark.svg").exists()  # no brand, so no mark
    assert out / "index.html" in report.pages


def test_the_project_templates_come_first_and_each_page_is_written(project):
    out = project / "dist"
    site_build.build(out, templates=project / "templates")
    assert "<p>About A Project.</p>" in (out / "about.html").read_text()
    assert not (out / "_part.html").exists()  # a part, not a page
    assert site_build.own_pages(project / "templates") == ["about.html", "index.html"]
    assert site_build.own_pages(project / "missing") == ["index.html"]


def test_the_command_line_builds_from_the_folders_it_finds(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    site_build.main(["--output-dir", "site"])  # no papers/ or posts/ folder here
    assert "pages to site" in capsys.readouterr().out
    assert (project / "site" / "about.html").exists()  # templates/ is the default


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # runpy: the module is imported already
def test_the_module_runs_as_a_script(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["build", "--output-dir", "out"])
    runpy.run_module("labinacan.site.build", run_name="__main__")
    assert (project / "out" / "index.html").exists()


def test_the_version_is_empty_for_a_project_that_is_not_installed(project):
    assert site_build._version() == ""  # "a-project" is no installed distribution


def test_the_accent_gives_the_stylesheet_its_brand_colours(tmp_path, restore):  # noqa: F811
    (tmp_path / "pyproject.toml").write_text(PYPROJECT + ACCENT)
    cfg.use(tmp_path / "pyproject.toml")
    assert accent_colours() == {"brand-400": "#34d399", "brand-700": "#047857"}
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)  # a brand with no accent
    cfg.use(tmp_path / "pyproject.toml")
    assert accent_colours() == {}


def test_a_project_with_a_brand_gets_its_images(tmp_path, restore, monkeypatch):  # noqa: F811
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    cfg.use(tmp_path / "pyproject.toml")
    drawn = tmp_path / "dist" / "assets" / "mark.svg"
    monkeypatch.setattr(site_build, "write_brand", lambda out: [drawn])  # the package isn't real
    report = site_build.build(tmp_path / "dist")
    assert drawn in report.pages


def test_a_project_with_no_brand_has_no_accent(project):
    assert accent_colours() == {}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # runpy: the module is imported already
def test_the_stylesheet_module_runs_as_a_script(project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["css", "--help"])  # argparse exits before any build
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("labinacan.site.css", run_name="__main__")
    assert exit_.value.code == 0 and "--site" in capsys.readouterr().out


def test_the_build_makes_a_nested_folder_and_builds_again_into_it(project):
    out = project / "a" / "b" / "dist"
    site_build.build(out)
    site_build.build(out)  # an existing folder is fine
    assert (out / "index.html").exists()


def test_the_build_hands_the_sections_what_it_was_given(project, monkeypatch):
    seen = {}

    def sections(render, report, own, **kwargs):
        seen.update(report=report, own=own, **kwargs)

    monkeypatch.setattr(site_build, "render_sections", sections)
    report = site_build.build(project / "dist", papers_dir=project / "p", posts_dir=project / "q")
    assert seen["report"] is report and len(seen["own"]) == 1
    assert seen["papers_dir"] == project / "p" and seen["posts_dir"] == project / "q"
    assert seen["pdfs"] is False and seen["research_index"] == {"explained": False}


def test_the_brand_is_drawn_into_the_site_folder(tmp_path, restore, monkeypatch):  # noqa: F811
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    cfg.use(tmp_path / "pyproject.toml")
    folders = []
    monkeypatch.setattr(site_build, "write_brand", lambda out: folders.append(out) or [])
    site_build.build(tmp_path / "dist")
    assert folders == [tmp_path / "dist"]


def test_the_templates_escape_html_and_strip_block_lines(tmp_path):
    (tmp_path / "t.html").write_text("<p>{{ x }}</p>\n  {% if x %}\nyes\n  {% endif %}\n")
    page = site_build.environment(tmp_path).get_template("t.html").render(x="<b>")
    assert page == "<p>&lt;b&gt;</p>\nyes\n"


def test_the_command_line_passes_its_defaults_and_only_the_folders_that_exist(project, monkeypatch):
    calls = []
    monkeypatch.setattr(
        site_build, "build", lambda *a: calls.append(a) or site_build.SiteReport(a[0])
    )
    monkeypatch.chdir(project)
    site_build.main([])
    assert calls[-1] == (site_build.Path("dist"), site_build.Path("templates"), None, None, False)
    (project / "papers").mkdir()
    (project / "posts").mkdir()
    site_build.main(["--pdfs", "--templates", "t"])
    out, templates, papers, posts, pdfs = calls[-1]
    assert (templates, papers, posts, pdfs) == (
        site_build.Path("t"),
        site_build.Path("papers"),
        site_build.Path("posts"),
        True,
    )
    assert all(isinstance(p, site_build.Path) for p in (out, templates, papers, posts))


def test_the_command_line_help_names_the_build_and_its_options(capsys):
    with pytest.raises(SystemExit):
        site_build.main(["--help"])
    text = capsys.readouterr().out
    assert site_build.__doc__.splitlines()[0] in text and "Print each paper's PDF." in text


def test_the_command_line_reads_the_folders_it_is_given(project, monkeypatch):
    calls = []
    monkeypatch.setattr(
        site_build, "build", lambda *a: calls.append(a) or site_build.SiteReport(a[0])
    )
    (project / "p").mkdir()
    (project / "q").mkdir()
    site_build.main(["--papers", str(project / "p"), "--posts", str(project / "q")])
    assert calls[-1][2:4] == (project / "p", project / "q")


def test_a_project_base_is_a_part_of_each_page_not_a_page(tmp_path):
    (tmp_path / "base.html").write_text("{% block content %}{% endblock %}")
    (tmp_path / "about.html").write_text("about")
    assert site_build.own_pages(tmp_path) == ["about.html", "index.html"]
