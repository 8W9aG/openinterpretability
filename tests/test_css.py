"""The site's stylesheet, built with Tailwind's CLI (openinterp/builder/css.py, #3282)."""

import json

import pytest

from openinterp.builder import css, palette


def test_the_config_holds_the_palette_and_the_fonts():
    text = css.config(["classes.txt"])
    assert text.startswith("module.exports = ") and text.endswith(";\n")
    conf = json.loads(text.removeprefix("module.exports = ").removesuffix(";\n"))
    assert conf["content"] == ["classes.txt"]
    assert conf["theme"]["extend"]["colors"] == palette.tailwind_colors()
    assert conf["theme"]["extend"]["fontFamily"] == css.FONTS


def test_candidates_come_from_class_attributes_and_scripts(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "p.html").write_text(
        '<p class="text-sm dark:text-slate-400">Body words stay out</p>'
        "<script>row.className = 'bg-status-verified px-2';</script>"
        "<SCRIPT type=module>x.className = 'mt-3'</SCRIPT >"
    )
    (tmp_path / "app.js").write_text('el.classList.add("underline")')
    found = css.candidates(tmp_path)
    assert {"text-sm", "dark:text-slate-400", "bg-status-verified", "px-2", "underline"} <= found
    assert "mt-3" in found  # a script in upper case, with a space before ">"
    assert "Body" not in found  # page text is not a class


def test_the_cli_must_match_its_pinned_hash(tmp_path):
    name, _ = css.BINARIES[("Linux", "x86_64")]
    (tmp_path / f"{name}-{css.VERSION}").write_bytes(b"not tailwind")
    with pytest.raises(SystemExit, match="SHA-256"):
        css.binary(tmp_path, ("Linux", "x86_64"))
    assert not (tmp_path / f"{name}-{css.VERSION}").exists()
    with pytest.raises(SystemExit, match="no Tailwind CLI"):
        css.binary(tmp_path, ("Plan9", "mips"))


def test_no_page_loads_the_tailwind_cdn():
    from pathlib import Path

    templates = Path(css.__file__).parent / "templates"
    for page in templates.rglob("*.html"):
        assert "cdn.tailwindcss.com" not in page.read_text(), page
    assert css.STYLESHEET in (templates / "base.html").read_text()


def test_search_loads_only_when_used():
    from pathlib import Path

    base = (Path(css.__file__).parent / "templates" / "base.html").read_text()
    assert '<script src="{{ root }}pagefind/pagefind-ui.js"' not in base  # not with every page
    assert 'id="search-stub"' in base and 'js.src = "{{ root }}pagefind/pagefind-ui.js"' in base
    assert "content-visibility: auto" in base  # long tables and maps are laid out near the screen
