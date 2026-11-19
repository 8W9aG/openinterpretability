"""The literature review on the site: a subsection of Research, one page per review (#3293).

The reviews are the Markdown files in ``docs/literature/``. ``README.md`` is their index,
and each other file is one paper or post. Each becomes a page under
``research/literature/``, rendered as the papers are
(:mod:`openinterpretability.builder.research`), with the same search for personal data.
Links between the reviews become links between the pages. Links to other repository files
point at GitHub. An install has no ``docs/`` folder, so it writes no pages.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.pii import redact_text
from openinterpretability.builder.research import _MARKDOWN, _plain

LITERATURE = Path(__file__).resolve().parents[2] / "docs" / "literature"  # absent in an install
SECTION = "research/literature"
MAP = "map.json"  # scripts/literature_map.py writes it (#3294)
STATUSES = ("used", "planned", "not applicable")  # the index table's status column
# A row of the index: "| [Title](name.md) (Authors, year) | status | why |".
_ROW = re.compile(r"^\| \[[^\]]*\]\(([\w-]+)\.md\)[^|]*\| ([a-z ]+) \| (.+?) \|\s*$", re.M)


@dataclass(frozen=True)
class Review:
    """One page of the literature review."""

    name: str  # the file's name without .md; README is the index
    title: str
    body: str  # rendered HTML without the title

    @property
    def href(self) -> str:
        """The page, relative to the site root."""
        page = "index" if self.name == "README" else self.name
        return f"{SECTION}/{page}.html"


def _rewrite(tokens, repo_url: str) -> None:
    """Point links at the review's pages, and links to other repository files at GitHub."""
    for token in tokens:
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if not href or re.match(r"[a-z]+:|#", href):
                continue
            target, _, fragment = href.partition("#")
            anchor = f"#{fragment}" if fragment else ""
            if "/" not in target and target.endswith(".md"):
                page = "index" if target == "README.md" else target[: -len(".md")]
                child.attrSet("href", f"{page}.html{anchor}")
            else:
                path = posixpath.normpath(posixpath.join("docs/literature", target))
                child.attrSet("href", f"{repo_url}/blob/main/{path}{anchor}")


def _review(path: Path, repo_url: str) -> Review:
    """One file, rendered: its level 1 title, and the rest as HTML."""
    tokens = _MARKDOWN.parse(redact_text(path.read_text(encoding="utf-8")))
    _rewrite(tokens, repo_url)
    start, title = 0, path.stem
    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
        title, start = _plain(tokens[1]), 3
    body = _MARKDOWN.renderer.render(tokens[start:], _MARKDOWN.options, {})
    return Review(path.stem, title, body)


def load_reviews(repo_url: str, folder: Path = LITERATURE) -> list[Review]:
    """The index first, then each review in file-name order; none without the folder."""
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.md"), key=lambda p: (p.name != "README.md", p.name))
    return [_review(path, repo_url) for path in files]


def map_text(path: Path) -> str | None:
    """What the map embeds for one review: its title and *Summary* section, or None.

    The index and the queue have no *Summary* section, so they are left off the map.
    """
    text = path.read_text(encoding="utf-8")
    title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), path.stem)
    found = re.search(r"^## Summary\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return f"{title}\n\n{found.group(1).strip()}" if found else None


def review_rows(folder: Path) -> dict[str, tuple[str, str]]:
    """Each review's (status, reason) from the index table of ``folder``."""
    index = folder / "README.md"
    text = index.read_text(encoding="utf-8") if index.exists() else ""
    return {name: (status.strip(), why.strip()) for name, status, why in _ROW.findall(text)}


def statuses(folder: Path) -> dict[str, str]:
    """Each review's status from the index table: used, planned or not applicable."""
    return {n: s for n, (s, _) in review_rows(folder).items() if s in STATUSES}


_CREDIT = re.compile(r"^\| \[[^\]]*\]\(([\w-]+)\.md\) \(([^|]+?)\) \|", re.M)


def credits(folder: Path) -> dict[str, str]:
    """Each review's "Authors, year", from the index table, for the map's cards."""
    index = folder / "README.md"
    return dict(_CREDIT.findall(index.read_text(encoding="utf-8"))) if index.exists() else {}


def sections(folder: Path) -> dict[str, str]:
    """The ``##`` section of the index that lists each review: its topic (#4577)."""
    index = folder / "README.md"
    out, section = {}, None
    for line in index.read_text(encoding="utf-8").splitlines() if index.exists() else []:
        if line.startswith("## "):
            section = line[3:].strip()
        elif (found := _ROW.match(line)) and section:
            out.setdefault(found.group(1), section)
    return out


def render_reviews(render, reviews: list[Review], folder: Path = LITERATURE) -> None:
    """Write one page per review, and the index with the map of the reviews."""
    from openinterpretability.builder.literature_map_view import (
        HEIGHT,
        WIDTH,
        literature_map,
        map_view,
    )

    points, missing = literature_map(folder, reviews) if reviews else ([], [])
    view = map_view(folder, points) if points else {"queued": [], "callouts": [], "badges": []}
    for review in reviews:
        extra = {"points": points, "missing": missing, "map_size": (WIDTH, HEIGHT), **view}
        extra = extra if review.name == "README" else {}
        render(review.href, "literature.html", review=review, **extra)
