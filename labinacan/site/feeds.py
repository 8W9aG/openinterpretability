"""A site's sitemap and its ``robots.txt``, from the pages that the build wrote (#3285, #6628).

Each file comes from the pages, never the clock. So an unchanged site rebuilds them byte for
byte, and the deploy uploads (and purges) nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

# The sitemap protocol allows at most this many URLs in one file.
SITEMAP_LIMIT = 50_000
# Any robots meta that says noindex: a redirect's, or a draft post's "noindex, nofollow".
NOINDEX = re.compile(r'<meta name="robots" content="[^"]*\bnoindex\b')


def sitemap(pages: Sequence[Path], output_dir: Path, site_url: str) -> str:
    """The sitemap of ``pages``, the HTML files the build wrote, without the redirects.

    Args:
        pages: The files that the build wrote.
        output_dir: The site's output folder.
        site_url: The site's URL, with a trailing slash.

    Returns:
        The sitemap's XML.

    Raises:
        ValueError: The pages are more than one sitemap holds (:data:`SITEMAP_LIMIT`).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "a.html"
        ...     _ = page.write_text("<p>A")
        ...     "<loc>https://x.org/a.html</loc>" in sitemap([page], Path(tmp), "https://x.org/")
        True
    """
    urls = []
    for page in sorted({p for p in pages if p.suffix == ".html"}):
        if NOINDEX.search(page.read_bytes().decode(errors="replace")):  # UTF-8, any locale
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


def write_site_files(output_dir: Path, pages: Sequence[Path], site_url: str) -> None:
    """Write ``sitemap.xml`` and ``robots.txt`` in ``output_dir``.

    Args:
        output_dir: The site's output folder.
        pages: The files that the build wrote.
        site_url: The site's URL, with a trailing slash.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_site_files(Path(tmp), [], "https://x.org/")
        ...     sorted(p.name for p in Path(tmp).iterdir())
        ['robots.txt', 'sitemap.xml']
    """
    (output_dir / "sitemap.xml").write_bytes(sitemap(pages, output_dir, site_url).encode())
    (output_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {site_url}sitemap.xml\n"
    )
