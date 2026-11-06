"""Publish the papers of ``papers/`` as the site's Research section.

``papers/README.md`` lists the papers, and its order is the order of the section's index. A
paper is a folder with a ``README.md``. The site renders that Markdown and copies ``media/``.
A folder that has an ``index.html`` is an interactive page, and the site copies it as it is.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from openinterp.builder.pii import redact_text

_LISTED = re.compile(r"\]\(([\w-]+)/README\.md\)")
_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")


@dataclass(frozen=True)
class Paper:
    """One entry of the Research section."""

    slug: str
    title: str
    summary: str
    body: str  # rendered HTML without the title, or empty for an interactive page
    hero: str | None  # the first figure, relative to the paper's folder
    interactive: bool

    @property
    def href(self) -> str:
        """The paper's page, relative to the section's index."""
        return f"{self.slug}/index.html"


def _plain(inline: Token) -> str:
    """Return the text of an inline token, without its marks."""
    kinds = {"text", "code_inline"}
    return "".join(child.content for child in inline.children or [] if child.type in kinds)


def _rewrite_links(tokens: list[Token], slug: str, repo_url: str) -> None:
    """Point links at the site's pages, and links to other repo files at GitHub."""
    for token in tokens:
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if not href or re.match(r"[a-z]+:|#", href):
                continue
            if href.endswith("README.md"):
                child.attrSet("href", href[: -len("README.md")] + "index.html")
            else:
                path = posixpath.normpath(posixpath.join("papers", slug, href))
                child.attrSet("href", f"{repo_url}/blob/main/{path}")


def _first_image(tokens: list[Token]) -> str | None:
    """Return the source of the first image, if there is one."""
    for token in tokens:
        for child in token.children or []:
            if child.type == "image":
                return child.attrGet("src")
    return None


def _read_paper(folder: Path, repo_url: str) -> Paper:
    """Render one paper. Its text passes the same search for personal data as all the site."""
    tokens = _MARKDOWN.parse(redact_text((folder / "README.md").read_text(encoding="utf-8")))
    if tokens[0].type != "heading_open" or tokens[0].tag != "h1":
        raise ValueError(f"{folder / 'README.md'} must start with a level 1 heading")
    first = next(i for i, token in enumerate(tokens) if token.type == "paragraph_open")
    _rewrite_links(tokens, folder.name, repo_url)
    interactive = (folder / "index.html").exists()
    return Paper(
        slug=folder.name,
        title=_plain(tokens[1]),
        summary=_plain(tokens[first + 1]),
        body="" if interactive else _MARKDOWN.renderer.render(tokens[3:], _MARKDOWN.options, {}),
        hero=_first_image(tokens),
        interactive=interactive,
    )


def load_papers(papers_dir: Path | None, repo_url: str) -> list[Paper]:
    """Return the listed papers in the listed order. Return none if there is no such folder."""
    index = Path(papers_dir) / "README.md" if papers_dir else None
    if index is None or not index.exists():
        return []
    slugs = _LISTED.findall(index.read_text(encoding="utf-8"))
    return [_read_paper(Path(papers_dir) / slug, repo_url) for slug in slugs]


def copy_assets(papers_dir: Path, paper: Paper, output_dir: Path) -> list[Path]:
    """Copy a paper's published files under ``research/``, and return the copied pages."""
    source = Path(papers_dir) / paper.slug
    target = output_dir / "research" / paper.slug
    target.mkdir(parents=True, exist_ok=True)
    if not paper.interactive:
        shutil.copytree(source / "media", target / "media", dirs_exist_ok=True)
        return []
    for path in sorted(source.iterdir()):
        if path.name != "README.md":
            shutil.copy(path, target / path.name)
    return [target / "index.html"]
