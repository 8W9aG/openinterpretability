"""The site's stylesheet, built with Tailwind's CLI (openinterpretability/builder/css.py, #3282)."""

import json

import pytest

from openinterpretability.builder import css, palette


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
        "<script>y.className = 'mb-4'</script\t\n bar>"
    )
    (tmp_path / "app.js").write_text('el.classList.add("underline")')
    found = css.candidates(tmp_path)
    assert {"text-sm", "dark:text-slate-400", "bg-status-verified", "px-2", "underline"} <= found
    assert "mt-3" in found  # a script in upper case, with a space before ">"
    assert "mb-4" in found  # an end tag with whitespace and text before ">"
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


def test_the_site_has_one_content_column():
    # #5726: one width, set in css.py. No template caps a part of the page at its own width,
    # and the widest fixed figures fit the column at their natural size.
    import re
    from pathlib import Path

    from openinterpretability.builder import charts, dashboard

    text = css.config([]).removeprefix("module.exports = ").removesuffix(";\n")
    assert json.loads(text)["theme"]["extend"]["maxWidth"] == {"column": "60rem", "page": "62rem"}
    templates = Path(css.__file__).parent / "templates"
    caps = re.compile(r"max-w-([3-7]xl|prose)\b")  # a tooltip may keep max-w-xs
    assert not [f.name for f in templates.glob("*.html") if caps.search(f.read_text())]
    column = css.COLUMN_REM * 16
    assert column >= 760 * charts.DIAGRAM_SCALE  # the About page's taxonomy
    assert column >= dashboard.CHART_WIDTH
