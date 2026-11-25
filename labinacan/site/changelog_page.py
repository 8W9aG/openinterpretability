"""The Changelog page: each released version of the package, from ``CHANGELOG.md`` (#3284).

Each ``## [<version>] - <date>`` section becomes a section of ``changelog.html`` with the
anchor ``v<version>``, so a page can link one version. The footer links the running
version. An empty ``[Unreleased]`` section is left out. The text is rendered as the papers
are (:mod:`labinacan.site.research`). Links to repo files point at GitHub,
and the same search for personal data runs on the text.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from labinacan.site.pii import redact_text
from labinacan.site.research import _MARKDOWN

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"  # absent in an install
_HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?\s*$", re.M)


@dataclass(frozen=True)
class Version:
    """One section of the changelog."""

    name: str  # 0.0.21, or Unreleased
    date: str | None
    body: str  # rendered HTML

    @property
    def anchor(self) -> str:
        """``v0.0.21``, or ``unreleased``: the section's id on the page."""
        return "unreleased" if self.name == "Unreleased" else f"v{self.name}"


def _render(text: str, repo_url: str) -> str:
    """Markdown to HTML, with links to repo files pointed at GitHub."""
    tokens = _MARKDOWN.parse(redact_text(text))
    for token in tokens:
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if href and not re.match(r"[a-z]+:|#", href):
                child.attrSet("href", f"{repo_url}/blob/main/{posixpath.normpath(href)}")
    return _MARKDOWN.renderer.render(tokens, _MARKDOWN.options, {})


def load_versions(repo_url: str, path: Path = CHANGELOG) -> list[Version]:
    """The changelog's versions, newest first; none when the file is missing."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    heads = list(_HEADING.finditer(text))
    versions = []
    for head, after in zip(heads, [*heads[1:], None], strict=True):
        body = text[head.end() : after.start() if after else len(text)].strip()
        if head["name"] == "Unreleased" and not body:
            continue
        versions.append(Version(head["name"], head["date"], _render(body, repo_url)))
    return versions
