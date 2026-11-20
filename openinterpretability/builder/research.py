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

import yaml  # a dependency of transformers and huggingface_hub
from markdown_it import MarkdownIt
from markdown_it.token import Token

from openinterpretability.builder.contents import anchor as section_anchor
from openinterpretability.builder.maths import MathsError, enable_maths
from openinterpretability.builder.pii import REMOVED_TEXT, find_spans, redact_text, redact_tokens
from openinterpretability.builder.social_media import drawn

_LISTED = re.compile(r"\]\(([\w-]+)/README\.md\)")
# An entry of papers/README.md: "- 2026-11-16: [Title](slug/README.md)". The date is optional.
# A paper's status line, under its abstract (#3150). A preprint names the issues still
# open on it. A published paper gives the day it was published.
STATUS = re.compile(
    r"^\*\*Status:\*\* (?:(Preprint)\. Work left: (.+)|Published (\d{4}-\d{2}-\d{2})\. (.+))$",
    re.M,
)
ADDENDA = "## Addenda"  # a published paper changes only below this heading


@dataclass(frozen=True)
class PaperStatus:
    """A paper's status: a preprint with work left, or published on a day."""

    published: str | None  # the day it was published, or None for a preprint
    note: str  # a preprint's work left, or what a published paper says about changes
    issues: tuple[int, ...] = ()  # the issues named in the note

    @property
    def preprint(self) -> bool:
        """True while work on the paper is left."""
        return self.published is None


def paper_status(text: str) -> PaperStatus | None:
    """The status line of a paper's Markdown, or None when it has none."""
    found = STATUS.search(text)
    if found is None:
        return None
    note = found.group(2) or found.group(4)
    outside = re.sub(r"\([^)]*\)", "", note)  # "(rewrite in the form of #4202)" names no work
    issues = tuple(int(n) for n in re.findall(r"#(\d+)", outside))
    return PaperStatus(found.group(3), note, issues)


# A post can be a draft instead: "- Draft: [Title](slug/README.md)" (#5529, #5551).
_ENTRY = re.compile(r"^- (?:(\d{4}-\d{2}-\d{2}|Draft): )?\[[^\]]*\]\(([\w-]+)/README\.md\)", re.M)
DRAFT = "Draft"  # the marker in place of a date
_MARKDOWN = enable_maths(MarkdownIt("commonmark", {"html": False}).enable("table"))
LONG_CELL = 40  # characters: a longer table cell is text, which may wrap (_prose.html, #5570)


def _mark_long_cells(state) -> None:
    """Give each table cell with more than :data:`LONG_CELL` characters the class ``wrap``.

    Prose tables keep cells on one line, so numbers and short values line up. A long cell is
    text, such as a literature table's reason, and ``_prose.html`` lets it wrap. CSS can't
    measure a cell's text, so the Markdown step marks it, for every page that this parser
    renders.

    Args:
        state: markdown-it's core state, whose tokens are changed in place.
    """
    tokens = state.tokens
    for i, token in enumerate(tokens[:-1]):
        if token.type in ("td_open", "th_open") and len(tokens[i + 1].content) > LONG_CELL:
            token.attrJoin("class", "wrap")


_MARKDOWN.core.ruler.push("long_cells", _mark_long_cells)


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
    status: PaperStatus | None = None  # preprint or published (#3150)
    draft: bool = False  # marked "Draft" in the index: built, but unlisted (#5551)
    hero_alt: str = ""  # the first figure's alt text, for X and link cards (#5534)
    social_image: str | None = None  # media/social.png, drawn from the first figure (#5534)
    social_video: str | None = None  # media/social.mp4, when that figure animates
    authors: tuple[str, ...] = ()  # from the folder's CITATION.cff, in order (#5260)

    @property
    def href(self) -> str:
        """The page, relative to the section's index. A draft's is under ``drafts/``.

        A draft has no listing page there, so only its URL leads to it (#5551).
        """
        return f"drafts/{self.slug}/index.html" if self.draft else f"{self.slug}/index.html"


def _plain(inline: Token) -> str:
    """Return the text of an inline token, without its marks."""
    kinds = {"text", "code_inline"}
    return "".join(child.content for child in inline.children or [] if child.type in kinds)


def _rewrite_links(tokens: list[Token], slug: str, repo_url: str, source: str = "papers") -> None:
    """Point links at the site's pages, and links to other repo files at GitHub.

    Args:
        tokens: The parsed Markdown, changed in place.
        slug: The folder of the paper or post.
        repo_url: The repository's URL, for links to files that the site doesn't publish.
        source: The repository folder that holds ``slug``: ``papers`` or ``posts``.
    """
    for token in tokens:
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if not href or re.match(r"[a-z]+:|#", href):
                continue
            if href.endswith("README.md"):
                child.attrSet("href", href[: -len("README.md")] + "index.html")
            else:
                path = posixpath.normpath(posixpath.join(source, slug, href))
                child.attrSet("href", f"{repo_url}/blob/main/{path}")


def _mark_sections(tokens: list[Token]) -> tuple[tuple[str, str], ...]:
    """Give each `##` heading an id, and return the (id, title) of each one, in order."""
    sections, taken = [], set()
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2":
            title = _plain(tokens[i + 1])
            anchor = section_anchor(title, taken)
            token.attrSet("id", anchor)
            sections.append((anchor, title))
    return tuple(sections)


def _first_image(tokens: list[Token]) -> tuple[str | None, str]:
    """Return the source and the alt text of the first image, or (None, "")."""
    for token in tokens:
        for child in token.children or []:
            if child.type == "image":
                return child.attrGet("src"), child.content
    return None, ""


def cff_authors(folder: Path) -> tuple[str, ...]:
    """The author names in ``folder/CITATION.cff``, in order, or none if it has no such file.

    A person is "given-names family-names", and a group is its ``name`` (#5260).

    Args:
        folder: The paper's folder.

    Returns:
        The names, or an empty tuple when the folder has no ``CITATION.cff``.
    """
    path = folder / "CITATION.cff"
    if not path.is_file():
        return ()
    people = yaml.safe_load(path.read_text(encoding="utf-8")).get("authors") or []
    return tuple(
        a.get("name") or " ".join(filter(None, (a.get("given-names"), a.get("family-names"))))
        for a in people
    )


def _read_paper(folder: Path, repo_url: str, source: str = "papers") -> Paper:
    """Render one paper or post. Its text passes the same search for personal data as all the site.

    Args:
        folder: The folder of the paper or post, with its ``README.md``.
        repo_url: The repository's URL.
        source: The repository folder that holds ``folder``: ``papers`` or ``posts``.

    Returns:
        The rendered entry, with no date: :func:`load_papers` adds it.

    Raises:
        ValueError: When the ``README.md`` doesn't start with a level 1 heading.
        MathsError: When its maths doesn't render.
    """
    interactive = (folder / "index.html").exists()
    text = redact_text((folder / "README.md").read_text(encoding="utf-8"))
    tokens = _MARKDOWN.parse(text)
    if tokens[0].type != "heading_open" or tokens[0].tag != "h1":
        raise ValueError(f"{folder / 'README.md'} must start with a level 1 heading")
    first = next(i for i, token in enumerate(tokens) if token.type == "paragraph_open")
    hero, alt = _first_image(tokens)
    image, video = drawn(folder)  # scripts/social_images.py draws them (#5534)
    _rewrite_links(tokens, folder.name, repo_url, source)
    sections = () if interactive else _mark_sections(tokens)
    try:
        body = "" if interactive else _MARKDOWN.renderer.render(tokens[3:], _MARKDOWN.options, {})
    except MathsError as exc:
        raise MathsError(f"{folder / 'README.md'}: {exc}") from None
    return Paper(
        slug=folder.name,
        title=_plain(tokens[1]),
        summary=_plain(tokens[first + 1]),
        body=body,
        hero=hero,
        hero_alt=alt,
        social_image=image,
        social_video=video,
        interactive=interactive,
        sections=sections,
        status=paper_status(text),
        authors=cff_authors(folder),
    )


def load_papers(papers_dir: Path | None, repo_url: str, source: str = "papers") -> list[Paper]:
    """Return the listed papers, newest first. Return none if there is no such folder.

    Each entry's date comes before it in papers/README.md (#3292). Papers of one date keep
    their order in the list, and an entry with no date goes last. The posts of ``posts/``
    have the same form, so this loads them too (#4650).

    Args:
        papers_dir: The folder whose ``README.md`` lists the entries, or None.
        repo_url: The repository's URL.
        source: The folder's name in the repository, for links to its files.

    Returns:
        The entries, newest first, or an empty list.
    """
    from dataclasses import replace

    index = Path(papers_dir) / "README.md" if papers_dir else None
    if index is None or not index.exists():
        return []
    entries = _ENTRY.findall(index.read_text(encoding="utf-8"))
    papers = [
        replace(
            _read_paper(Path(papers_dir) / slug, repo_url, source),
            published=None if date in ("", DRAFT) else date,
            draft=date == DRAFT,
        )
        for date, slug in entries
    ]
    return sorted(papers, key=lambda p: p.published or "", reverse=True)


VENDORED: dict[str, str] = {}
"""The vendored libraries that the site copies as is, by file name and SHA-256.

The site redacts every other paper file that is UTF-8 text. A changed library needs a new hash.
"""

_WEB_SUFFIXES = {".html", ".htm", ".xhtml", ".svg", ".js", ".mjs", ".css"}
_STRING_ARRAY = re.compile(r'\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\]')


def _in_number(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` is part of a longer number, for example ``0.1277362108``.

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


def copy_assets(
    papers_dir: Path, paper: Paper, output_dir: Path, section: str = "research"
) -> list[Path]:
    """Copy a paper's published files under ``section/``, and return the copied pages.

    Text files pass through :func:`redact_asset`, so that a takedown or the search for
    personal data covers them as it covers every other page.

    Args:
        papers_dir: The folder that holds the paper's folder.
        paper: The paper or post.
        output_dir: The site's output folder.
        section: The site folder of the section: ``research`` for papers, ``posts`` for posts.

    Returns:
        The interactive page that was copied, or an empty list.
    """
    source = Path(papers_dir) / paper.slug
    target = output_dir / section / posixpath.dirname(paper.href)
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
