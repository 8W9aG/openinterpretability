"""Pages outside the model tree: About, Changelog, Research and Posts (builder/sections.py)."""

from openinterpretability.builder import sections
from openinterpretability.builder.compiler import compile_site


def test_an_install_without_a_changelog_builds_no_changelog_page(tmp_path, monkeypatch):
    monkeypatch.setattr(sections, "load_versions", lambda repo_url: [])
    compile_site([], tmp_path)
    assert (tmp_path / "about.html").exists()
    assert not (tmp_path / "changelog.html").exists()


def test_an_interactive_post_is_copied_as_it_is_and_not_rendered(tmp_path):
    posts = tmp_path / "posts"
    folder = posts / "live"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("# Live post\n\nTry it.\n")
    (folder / "index.html").write_text("<!doctype html><p>own page")
    (posts / "README.md").write_text("# Posts\n\n- 2026-11-20: [Live post](live/README.md)\n")
    out = tmp_path / "dist"
    compile_site([], out, posts_dir=posts)
    assert (out / "posts" / "live" / "index.html").read_text() == "<!doctype html><p>own page"
    assert "Live post" in (out / "posts" / "index.html").read_text()


def test_research_reads_its_papers_with_the_repo_url_and_makes_each_folder(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace

    from labinacan.site import paper_pdf

    paper, seen, pages = SimpleNamespace(slug="p", href="p/index.html", interactive=False), {}, []
    monkeypatch.setattr(sections, "load_papers", lambda d, url: seen.update(url=url) or [paper])
    monkeypatch.setattr(sections, "copy_assets", lambda *a: [])
    monkeypatch.setattr(paper_pdf, "write_pdf", lambda env, folder, *a: seen.update(pdf=folder))

    class Render:  # writes no file, so research/ doesn't exist before the paper's folder
        common, output_dir, env = {"repo_url": "https://x.org/r"}, tmp_path / "out", None

        def __call__(self, href, template, **kw):
            pages.append((href, template))

    found = sections._render_research(Render(), tmp_path, SimpleNamespace(pages=[]), False, 0, 0)
    assert found == [paper] and seen["url"] == "https://x.org/r"  # repo links in the papers
    assert seen["pdf"] == tmp_path / "out" / "research" / "p" and seen["pdf"].is_dir()
    assert pages == [
        ("research/index.html", "research.html"),
        ("research/p/index.html", "paper.html"),
    ]
    assert capsys.readouterr().out == ""  # no PDFs asked for: no warning about Chrome


def test_research_tells_the_index_what_exists_and_warns_without_chrome(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace

    from labinacan.site import paper_pdf

    monkeypatch.setattr(sections, "load_papers", lambda d, url: [])
    monkeypatch.setattr(paper_pdf, "chrome", lambda: None)
    calls = []

    class Render:
        common, output_dir, env = {"repo_url": "r"}, tmp_path, None

        def __call__(self, href, template, **kw):
            calls.append((href, template, kw))

    sections._render_research(Render(), tmp_path, SimpleNamespace(pages=[]), True, True, False)
    assert calls == [("research/index.html", "research.html",
                      {"papers": [], "explained": True, "reviewed": False})]  # fmt: skip
    warning = "No Chrome found ($OPENINTERPRETABILITY_CHROME): the papers get no PDFs.\n"
    assert capsys.readouterr().out == warning
    monkeypatch.setattr(paper_pdf, "chrome", lambda: "/bin/chrome")
    sections._render_research(Render(), tmp_path, SimpleNamespace(pages=[]), True, True, False)
    assert capsys.readouterr().out == ""  # a browser: no warning
