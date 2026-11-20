"""The site's sitemap, its feed of new VERIFIED units and its ``robots.txt`` (#3285).

Each file comes from the data, never the clock. So an unchanged registry rebuilds them
byte for byte, and the deploy uploads (and purges) nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from xml.sax.saxutils import escape

from openinterpretability.builder.experiment import Experiment

FEED_ENTRIES = 50
# The sitemap protocol allows at most this many URLs in one file.
SITEMAP_LIMIT = 50_000
# Any robots meta that says noindex: a redirect's, or a draft post's "noindex, nofollow".
NOINDEX = re.compile(r'<meta name="robots" content="[^"]*\bnoindex\b')


def sitemap(pages: Sequence[Path], output_dir: Path, site_url: str) -> str:
    """The sitemap of ``pages``, the HTML files the build wrote, without the redirects."""
    urls = []
    for page in sorted({p for p in pages if p.suffix == ".html"}):
        if NOINDEX.search(page.read_text(encoding="utf-8", errors="replace")):
            continue  # a redirect to a page that moved, or a draft post (#5551)
        rel = page.relative_to(output_dir).as_posix()
        urls.append(site_url + ("" if rel == "index.html" else rel))
    if len(urls) > SITEMAP_LIMIT:
        raise ValueError(f"{len(urls)} pages: split the sitemap, which holds {SITEMAP_LIMIT}")
    body = "".join(f"  <url><loc>{escape(url)}</loc></url>\n" for url in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}</urlset>\n"
    )


def _when(experiment: Experiment) -> str:
    """The verdict's time, in the form that Atom takes."""
    return experiment.record.verdict.timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def feed(experiments: Sequence[Experiment], site_url: str, site_title: str) -> str:
    """An Atom feed of the newest VERIFIED experiments that no erratum sets aside."""
    verified = [e for e in experiments if e.record.verdict.status == "VERIFIED" and not e.set_aside]
    newest = sorted(verified, key=lambda e: (_when(e), e.href), reverse=True)[:FEED_ENTRIES]
    entries = []
    for e in newest:
        url = escape(site_url + e.href)
        slug = escape(e.record.submission.identifier.slug)
        entries.append(
            f"  <entry>\n    <title>{slug} VERIFIED</title>\n"
            f'    <link href="{url}"/>\n    <id>{url}</id>\n'
            f"    <updated>{_when(e)}</updated>\n"
            f"    <summary>{escape(e.record.submission.natural_language_explanation)}</summary>\n"
            "  </entry>\n"
        )
    updated = _when(newest[0]) if newest else "1970-01-01T00:00:00Z"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        f"  <title>{escape(site_title)}: new VERIFIED units</title>\n"
        f'  <link href="{escape(site_url)}"/>\n'
        f'  <link rel="self" href="{escape(site_url)}feed.xml"/>\n'
        f"  <id>{escape(site_url)}feed.xml</id>\n"
        f"  <updated>{updated}</updated>\n"
        f"{''.join(entries)}</feed>\n"
    )


def write_feeds(report, experiments: Sequence[Experiment], common: dict) -> None:
    """Write ``sitemap.xml``, ``feed.xml`` and ``robots.txt`` in the build's output dir."""
    out, url = report.output_dir, common["site_url"]
    (out / "sitemap.xml").write_text(sitemap(report.pages, out, url), encoding="utf-8")
    (out / "feed.xml").write_text(feed(experiments, url, common["site_title"]), encoding="utf-8")
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {url}sitemap.xml\n")
