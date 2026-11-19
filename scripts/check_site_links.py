"""Check every internal link and anchor in the built site (issue #3170).

``make lint`` checks the relative links in hand-written Markdown
(``scripts/check_markdown.py``). The site is generated, and it cross-links every model,
layer, unit, experiment and explanation. This script reads each ``.html`` file under the
site folder and checks each ``href`` and ``src`` that stays inside the site:

- the file it names exists (a link that ends in ``/`` names that folder's ``index.html``);
- the ``#anchor`` it names is an ``id`` (or an ``<a name>``) in that file.

Links with a scheme (``https:``, ``mailto:``, ``data:``) are external. The weekly link
workflow checks the external links in the Markdown with lychee
(``.github/workflows/links.yml``). Run it after a build::

    uv run python scripts/check_site_links.py dist
"""

from __future__ import annotations

import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

SHOWN = 40  # broken links printed; the count is always printed


class _Page(HTMLParser):
    """The links and the anchor targets of one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        for name in ("href", "src"):
            if found.get(name):
                self.links.append(found[name] or "")
        if found.get("id"):
            self.ids.add(found["id"] or "")
        if tag == "a" and found.get("name"):
            self.ids.add(found["name"] or "")


def read(page: Path) -> _Page:
    """Parse ``page``."""
    parsed = _Page()
    parsed.feed(page.read_text(encoding="utf-8", errors="replace"))
    return parsed


def target(site: Path, page: Path, link: str) -> tuple[Path, str] | None:
    """The file and the anchor that ``link`` on ``page`` names, or None for an external one."""
    parts = urlsplit(link)
    if parts.scheme or parts.netloc or link.startswith("//"):
        return None
    if not parts.path:
        return page, unquote(parts.fragment)
    path = unquote(parts.path)
    base = site if path.startswith("/") else page.parent
    found = (base / path.lstrip("/")).resolve()
    if path.endswith("/") or found.is_dir():
        found = found / "index.html"
    return found, unquote(parts.fragment)


def broken_links(site: Path) -> list[tuple[str, str, str]]:
    """``(page, link, why)`` for each internal link under ``site`` that leads nowhere."""
    site = site.resolve()
    pages = sorted(site.rglob("*.html"))
    parsed = {p.resolve(): read(p) for p in pages}
    out = []
    for page, found in parsed.items():
        for link in found.links:
            hit = target(site, page, link)
            if hit is None:
                continue
            path, anchor = hit
            name = str(page.relative_to(site))
            if site not in path.parents and path != site:
                out.append((name, link, "leaves the site"))
            elif not path.exists():
                out.append((name, link, "no such file"))
            elif anchor and path.suffix == ".html":
                ids = parsed[path].ids if path in parsed else read(path).ids
                if anchor not in ids:
                    out.append((name, link, f"no #{anchor} in it"))
    return out


def main(argv: list[str]) -> int:
    """Print the broken links under the site folder in ``argv``, and fail if any."""
    if len(argv) != 1:
        print("usage: check_site_links.py SITE_DIR", file=sys.stderr)
        return 2
    site = Path(argv[0])
    found = broken_links(site)
    pages = sum(1 for _ in site.rglob("*.html"))
    if not found:
        print(f"{pages} pages: every internal link and anchor resolves.")
        return 0
    by_link: dict[tuple[str, str], list[str]] = defaultdict(list)
    for page, link, why in found:
        by_link[(link, why)].append(page)
    print(f"{pages} pages: {len(found)} broken internal links ({len(by_link)} distinct):")
    for (link, why), where in sorted(by_link.items(), key=lambda kv: -len(kv[1]))[:SHOWN]:
        print(f"  {link} ({why}): on {len(where)} page(s), e.g. {where[0]}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
