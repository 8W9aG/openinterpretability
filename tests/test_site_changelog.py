"""This site's Changelog page and its footer link (#6631: the template doesn't take it)."""

from openinterpretability.builder.compiler import compile_site


def test_the_site_has_the_page_and_the_footer_links_the_running_version(tmp_path, record):
    from openinterpretability import __version__

    compile_site([record], tmp_path)
    page = (tmp_path / "changelog.html").read_text(encoding="utf-8")
    assert f'id="v{__version__}"' in page and "Ships with protocol version" in page
    footer = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert f"changelog.html#v{__version__}" in footer
    assert "changelog.html" in (tmp_path / "about.html").read_text(encoding="utf-8")
