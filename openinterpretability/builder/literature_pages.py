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


def render_reviews(render, reviews: list[Review]) -> None:
    """Write one page per review, the index included."""
    for review in reviews:
        render(review.href, "literature.html", review=review)
