"""Publish the papers of ``papers/`` as the site's Research section.

``papers/README.md`` lists the papers, and its order is the order of the section's index. A
paper is a folder with a ``README.md``. The site renders that Markdown and copies ``media/``.
A folder that has an ``index.html`` is an interactive page, and the site copies its files.
Copied text files lose personal data and taken-down passages, as every page does.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from openinterpretability.builder.pii import REMOVED_TEXT, find_spans, redact_text, redact_tokens

_LISTED = re.compile(r"\]\(([\w-]+)/README\.md\)")
# An entry of papers/README.md: "- 2026-11-16: [Title](slug/README.md)". The date is optional.
_ENTRY = re.compile(r"^- (?:(\d{4}-\d{2}-\d{2}): )?\[[^\]]*\]\(([\w-]+)/README\.md\)", re.M)
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
    sections: tuple[tuple[str, str], ...] = ()  # (id, title) of each `##` heading, in order
    published: str | None = None  # the date before its entry in papers/README.md

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


def _anchor(title: str, taken: set[str]) -> str:
    """A stable id for a section, from its title: lower case, hyphens, and unique on the page."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    anchor, n = base, 2
    while anchor in taken:
        anchor, n = f"{base}-{n}", n + 1
    taken.add(anchor)
    return anchor


def _mark_sections(tokens: list[Token]) -> tuple[tuple[str, str], ...]:
    """Give each `##` heading an id, and return the (id, title) of each one, in order."""
    sections, taken = [], set()
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2":
            title = _plain(tokens[i + 1])
            anchor = _anchor(title, taken)
            token.attrSet("id", anchor)
            sections.append((anchor, title))
    return tuple(sections)


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
    sections = () if interactive else _mark_sections(tokens)
    return Paper(
        slug=folder.name,
        title=_plain(tokens[1]),
        summary=_plain(tokens[first + 1]),
        body="" if interactive else _MARKDOWN.renderer.render(tokens[3:], _MARKDOWN.options, {}),
        hero=_first_image(tokens),
        interactive=interactive,
        sections=sections,
    )


def load_papers(papers_dir: Path | None, repo_url: str) -> list[Paper]:
    """Return the listed papers, newest first. Return none if there is no such folder.

    Each entry's date comes before it in papers/README.md (#3292). Papers of one date keep
    their order in the list, and an entry with no date goes last.
    """
    from dataclasses import replace

    index = Path(papers_dir) / "README.md" if papers_dir else None
    if index is None or not index.exists():
        return []
    entries = _ENTRY.findall(index.read_text(encoding="utf-8"))
    papers = [
        replace(_read_paper(Path(papers_dir) / slug, repo_url), published=date or None)
        for date, slug in entries
    ]
    return sorted(papers, key=lambda p: p.published or "", reverse=True)


VENDORED = {
    "three.min.js": "9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2",
}
"""The vendored libraries that the site copies as is, by file name and SHA-256.

The site redacts every other paper file that is UTF-8 text. A changed library needs a new hash.
"""

_WEB_SUFFIXES = {".html", ".htm", ".xhtml", ".svg", ".js", ".mjs", ".css"}
_STRING_ARRAY = re.compile(r'\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\]')


def _in_number(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` is part of a longer number, e.g. ``0.1277362108``.

    Data and code files hold many long numbers, and the phone and card searches match
    their digits. A match inside a number is not personal data.
    """
    before = text[start - 1] if start else ""
    after = text[end : end + 2]
    return (
        before in set(".0123456789")
        or after[:1].isdigit()
        or (after[:1] == "." and after[1:2].isdigit())
    )


def redact_asset(text: str) -> str:
    """Return a paper's data or code file with personal data and taken-down text removed.

    It is :func:`redact_text` (:mod:`openinterpretability.builder.pii`) without the matches inside
    numbers, which would break the file and hold no personal data. Before that, it reads each
    JSON array of strings as a token list (:func:`redact_tokens`), because a match can span
    several tokens.
    """
    text = _STRING_ARRAY.sub(_redact_array, text)
    spans = [s for s in find_spans(text) if not _in_number(text, *s)]
    for start, end in reversed(spans):
        text = text[:start] + REMOVED_TEXT + text[end:]
    return text


def _redact_array(match: re.Match[str]) -> str:
    """Return a JSON array of strings with :func:`redact_tokens` applied to it."""
    try:
        tokens = json.loads(match.group())
    except ValueError:
        return match.group()
    redacted = redact_tokens(tokens)
    return match.group() if redacted == tokens else json.dumps(redacted, ensure_ascii=False)


def _copy(path: Path, target: Path) -> None:
    """Copy one file, redacted if it is UTF-8 text.

    A library in :data:`VENDORED` with the pinned hash is copied as is. A file that is not
    UTF-8 is copied as is, except a web page or script, which stops the build.
    """
    data = path.read_bytes()
    if VENDORED.get(path.name) == hashlib.sha256(data).hexdigest():
        shutil.copy(path, target)
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if path.suffix.lower() in _WEB_SUFFIXES:
            raise ValueError(f"{path}: a web page or script must be UTF-8 text") from None
        shutil.copy(path, target)
        return
    target.write_text(redact_asset(text), encoding="utf-8")


def copy_assets(papers_dir: Path, paper: Paper, output_dir: Path) -> list[Path]:
    """Copy a paper's published files under ``research/``, and return the copied pages.

    Text files pass through :func:`redact_asset`, so that a takedown or the search for
    personal data covers them as it covers every other page.
    """
    source = Path(papers_dir) / paper.slug
    target = output_dir / "research" / paper.slug
    target.mkdir(parents=True, exist_ok=True)
    if not paper.interactive:
        for path in sorted((source / "media").rglob("*")):
            if path.is_file():
                out = target / "media" / path.relative_to(source / "media")
                out.parent.mkdir(parents=True, exist_ok=True)
                _copy(path, out)
        return []
    for path in sorted(source.iterdir()):
        if path.name != "README.md" and path.is_file():
            _copy(path, target / path.name)
    return [target / "index.html"]
