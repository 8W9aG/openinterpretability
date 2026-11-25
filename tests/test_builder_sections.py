"""Pages outside the model tree: About, Changelog, Research and Posts (builder/sections.py)."""

from openinterpretability.builder.compiler import compile_site


def test_an_install_without_a_changelog_builds_no_changelog_page(tmp_path, monkeypatch):
    from labinacan.site import sections as generic

    monkeypatch.setattr(generic, "load_versions", lambda repo_url: [])
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


def test_this_repos_own_sections_are_about_and_the_explanations(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openinterpretability import DISTRIBUTION
    from openinterpretability.builder import sections
    from openinterpretability.core.licenses import DATA_TERMS
    from openinterpretability.core.schemas import ReviewPolicy

    seen = {}
    monkeypatch.setattr(sections, "metadata", lambda dist: {"License-Expression": f"L:{dist}"})
    monkeypatch.setattr(sections, "taxonomy_rows", lambda url: f"taxonomy:{url}")
    monkeypatch.setattr(sections, "write_brand", lambda out: [f"brand:{out}"])
    explain = lambda r, views, units: seen.update(explained=(r, views, units))  # noqa: E731
    monkeypatch.setattr(sections, "render_explanations", explain)

    def common(render, report, own, **kw):
        seen["kw"] = kw
        for section in own:
            section(render)

    monkeypatch.setattr(sections, "render_sections", common)

    class Render:
        common, output_dir, pages = {"repo_url": "R"}, tmp_path, []

        def __call__(self, href, template, **kw):
            self.pages.append((href, template, kw))

    render, report = Render(), SimpleNamespace(pages=[])
    sections.render_reference_pages(render, tmp_path / "p", report, True, iter(["view"]),
                                    tmp_path / "q", {"u"})  # fmt: skip
    about = {"policy": ReviewPolicy(), "license": f"L:{DISTRIBUTION}",
             "taxonomy": "taxonomy:R", **DATA_TERMS}  # fmt: skip
    assert render.pages == [("about.html", "about.html", about)]
    assert report.pages == [f"brand:{tmp_path}"]
    assert seen["explained"] == (render, ["view"], {"u"})
    assert seen["kw"] == {"papers_dir": tmp_path / "p", "posts_dir": tmp_path / "q", "pdfs": True,
                          "research_index": {"explained": True}}  # fmt: skip
    sections.render_reference_pages(render, None, report)  # the defaults: no PDFs, no explanations
    assert seen["kw"] == {"papers_dir": None, "posts_dir": None, "pdfs": False,
                          "research_index": {"explained": False}}  # fmt: skip
