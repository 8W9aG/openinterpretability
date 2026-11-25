"""The site's feed of new VERIFIED units, beside labinacan's sitemap and ``robots.txt`` (#3285).

The feed comes from the records, never the clock. So an unchanged registry rebuilds it byte
for byte, and the deploy uploads (and purges) nothing. ``labinacan/site/feeds.py`` writes
the sitemap and ``robots.txt`` (#6628).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from xml.sax.saxutils import escape

from labinacan.site.feeds import write_site_files
from openinterpretability.builder.experiment import Experiment

FEED_ENTRIES = 50


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
    write_site_files(out, report.pages, url)
    (out / "feed.xml").write_bytes(feed(experiments, url, common["site_title"]).encode())
