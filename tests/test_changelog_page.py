"""The Changelog page: one section per released version (builder/changelog_page.py, #3284)."""

from openinterpretability.builder.changelog_page import load_versions
from openinterpretability.builder.compiler import compile_site

TEXT = """# Changelog

Intro.

## [Unreleased]

## [0.0.2] - 2026-11-16

Ships with protocol version **0.9**.

### Added

- A thing, see [the README](changelog.d/README.md) and [a site](https://example.org).

## [0.0.1] - 2026-11-01

- The first.
"""


def test_each_version_is_a_section_and_an_empty_unreleased_is_left_out(tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(TEXT)
    new, old = load_versions("https://github.com/o/r", path)
    assert (new.name, new.date, new.anchor, old.anchor) == (
        "0.0.2",
        "2026-11-16",
        "v0.0.2",
        "v0.0.1",
    )
    assert "<strong>0.9</strong>" in new.body and "<h3>Added</h3>" in new.body
    assert 'href="https://github.com/o/r/blob/main/changelog.d/README.md"' in new.body
    assert 'href="https://example.org"' in new.body
    path.write_text(TEXT.replace("## [Unreleased]\n", "## [Unreleased]\n\n- Coming.\n"))
    assert load_versions("u", path)[0].anchor == "unreleased"
    assert load_versions("u", tmp_path / "missing.md") == []


def test_the_site_has_the_page_and_the_footer_links_the_running_version(tmp_path, record):
    from openinterpretability import __version__

    compile_site([record], tmp_path)
    page = (tmp_path / "changelog.html").read_text(encoding="utf-8")
    assert f'id="v{__version__}"' in page and "Ships with protocol version" in page
    footer = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert f"changelog.html#v{__version__}" in footer
    assert "changelog.html" in (tmp_path / "about.html").read_text(encoding="utf-8")
