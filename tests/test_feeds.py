"""The site's sitemap, feed and robots.txt (``openinterpretability/builder/feeds.py``, #3285)."""

import xml.etree.ElementTree as ET

from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.feeds import NOINDEX
from openinterpretability.core.project import site_url
from tests.test_compiler import experiment

ATOM = "{http://www.w3.org/2005/Atom}"
SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def _records(record):
    return [
        experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0), hours=2),
        experiment(record, idx=9, status="VERIFIED", counts=(10, 0, 5, 0), hours=5, text="Newer."),
        experiment(record, idx=4, status="REJECTED", counts=(3, 3, 1, 3), text="Fires on dates."),
    ]


def test_the_sitemap_lists_the_pages_but_not_the_redirects(tmp_path, record):
    compile_site(_records(record), tmp_path)
    locs = [e.text for e in ET.parse(tmp_path / "sitemap.xml").getroot().iter(f"{SITEMAP}loc")]
    assert site_url() in locs  # the home page, at the site's root
    assert any(loc.endswith("/L3/N117.html") for loc in locs)  # the unit page
    assert any("/L3/N117-positive/" in loc for loc in locs)  # the experiment page
    for loc in locs:
        page = tmp_path / (loc.removeprefix(site_url()) or "index.html")
        assert not NOINDEX.search(page.read_text(encoding="utf-8"))
    assert f"Sitemap: {site_url()}sitemap.xml" in (tmp_path / "robots.txt").read_text()


def test_the_feed_holds_only_verified_experiments_newest_first(tmp_path, record):
    compile_site(_records(record), tmp_path)
    root = ET.parse(tmp_path / "feed.xml").getroot()
    entries = root.findall(f"{ATOM}entry")
    titles = [e.find(f"{ATOM}title").text for e in entries]
    assert len(entries) == 2 and titles[0].endswith("/L3/N9-positive VERIFIED")
    assert entries[0].find(f"{ATOM}summary").text == "Newer."
    assert root.find(f"{ATOM}updated").text == entries[0].find(f"{ATOM}updated").text
    link = entries[0].find(f"{ATOM}link").get("href")
    assert link.startswith(site_url()) and (tmp_path / link.removeprefix(site_url())).exists()


def test_an_unchanged_registry_rebuilds_the_same_files(tmp_path, record):
    for name in ("a", "b"):
        compile_site(_records(record), tmp_path / name)
    for file in ("sitemap.xml", "feed.xml", "robots.txt"):
        assert (tmp_path / "a" / file).read_bytes() == (tmp_path / "b" / file).read_bytes()


def test_every_page_links_the_feed(tmp_path, record):
    compile_site(_records(record), tmp_path)
    assert 'type="application/atom+xml"' in (tmp_path / "index.html").read_text()
