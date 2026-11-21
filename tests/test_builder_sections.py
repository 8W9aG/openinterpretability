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
