"""labinacan's sitemap and robots.txt: what they list, their form and their limit (#6628)."""

from labinacan.site import feeds


def _pages(tmp_path, *names, text="<p>page"):
    out = []
    for name in names:
        page = tmp_path / name
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text)
        out.append(page)
    return out


def test_the_sitemap_lists_each_page_once_in_order(tmp_path):
    pages = _pages(tmp_path, "index.html", "m/a.html")
    xml = feeds.sitemap([*pages, pages[0], tmp_path / "data.json"], tmp_path, "https://x.org/")
    assert xml == (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url><loc>https://x.org/</loc></url>\n"
        "  <url><loc>https://x.org/m/a.html</loc></url>\n"
        "</urlset>\n"
    )


def test_a_page_that_is_not_utf8_is_still_read_for_its_robots_meta(tmp_path):
    page = tmp_path / "old.html"
    page.write_bytes(b'\xff<meta name="robots" content="noindex">')  # one byte is not UTF-8
    assert "<loc>" not in feeds.sitemap([page], tmp_path, "https://x.org/")


def test_a_sitemap_holds_exactly_its_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(feeds, "SITEMAP_LIMIT", 2)
    pages = _pages(tmp_path, "a.html", "b.html")
    assert feeds.sitemap(pages, tmp_path, "https://x.org/").count("<loc>") == 2


def test_the_site_files_are_named_and_written_exactly(tmp_path):
    page = _pages(tmp_path, "index.html")
    feeds.write_site_files(tmp_path, page, "https://x.org/")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.html", "robots.txt", "sitemap.xml"]
    assert (tmp_path / "robots.txt").read_text() == (
        "User-agent: *\nAllow: /\nSitemap: https://x.org/sitemap.xml\n"
    )
    assert (tmp_path / "sitemap.xml").read_bytes() == feeds.sitemap(
        page, tmp_path, "https://x.org/"
    ).encode()
