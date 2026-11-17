"""Tables of contents on long pages, built from their headings (builder/contents.py, #3756)."""

import re

from openinterpretability.builder.contents import MIN_SECTIONS, add_contents, anchor


def _page(level, titles, extra=""):
    headings = "".join(f"<h{level}>{t}</h{level}><p>Text.</p>" for t in titles)
    main = f"<main><h1>Title</h1>{extra}{headings}</main>"
    return f"<html><body>{main}<footer><h2>Foot</h2></footer></body></html>"


def _links(page):
    nav = page[page.index('aria-label="Contents"') : page.index("</nav>")]
    return re.findall(r'href="#([^"]+)">([^<]*)<', nav)


def test_a_long_page_lists_its_top_level_sections_in_order():
    titles = ["Method", "Results", "Results", "Caveats &amp; limits", "Next"]
    page = add_contents(_page(2, titles))
    expected = [("method", "Method"), ("results", "Results"), ("results-2", "Results"),
                ("caveats-limits", "Caveats &amp; limits"), ("next", "Next")]  # fmt: skip
    assert _links(page) == expected
    for name, _ in expected:
        assert f'id="{name}"' in page
    assert page.count('aria-label="Contents"') == 2  # under the h1, and in the margin
    assert page.index('aria-label="Contents"') > page.index("</h1>")
    assert 'id="foot"' not in page  # only headings inside <main>


def test_the_top_level_is_the_highest_level_the_page_uses():
    page = add_contents(_page(3, [f"Part {i}" for i in range(MIN_SECTIONS)], "<h4>Inner</h4>"))
    assert [name for name, _ in _links(page)] == [f"part-{i}" for i in range(MIN_SECTIONS)]


def test_a_short_page_gets_ids_but_no_contents():
    page = add_contents(_page(2, ["One", "Two"]))
    assert 'aria-label="Contents"' not in page and 'id="one"' in page


def test_an_existing_id_is_kept_and_anchors_are_stable():
    titles = [f"S{i}" for i in range(MIN_SECTIONS)]
    page = _page(2, titles).replace("<h2>S0</h2>", '<h2 id="kept">S0</h2>')
    once = add_contents(page)
    assert _links(once)[0] == ("kept", "S0")
    assert add_contents(page) == once  # the same page builds the same anchors


def test_an_anchor_is_unique_and_safe():
    taken = set()
    assert [anchor(t, taken) for t in ("A b", "A b", "***")] == ["a-b", "a-b-2", "section"]
