"""This site's Changelog page and its footer link (#6631: the template doesn't take it)."""

from openinterpretability.builder.compiler import compile_site


def test_only_the_changelog_names_the_version_and_every_page_links_it(tmp_path, record):
    # A release changes the version, so a page that showed it would change at each release,
    # and the deploy would upload every page again (#7107).
    from openinterpretability import __version__

    compile_site([record], tmp_path)
    page = (tmp_path / "changelog.html").read_text(encoding="utf-8")
    assert f'id="v{__version__}"' in page and "Ships with protocol version" in page
    footer = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'href="changelog.html">Changelog</a>' in footer
    assert "changelog.html" in (tmp_path / "about.html").read_text(encoding="utf-8")
    named = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.html")
             if __version__ in p.read_text(encoding="utf-8")]  # fmt: skip
    assert named == ["changelog.html"]
