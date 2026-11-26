"""Find the project's own terms in a paper, where a reader outside the project meets them (#6851).

A paper is for readers outside the project, so it uses the field's terms or plain words.
``papers/terms.toml`` lists each project term with a regular expression and the plain phrase
to use. This module finds the terms in a paper's prose, abstract and captions, in its
figures' titles and subtitles, and in its X thread. Code, the References and the Reproduce
appendix are exempt, because commands and paths belong there.

``papers/terms_pending.txt`` lists the preprints that wait for a rewrite, one slug on each
line. The check skips them, and asks for a line to go once its paper has no term left. A
published paper is frozen (``write-paper``, *Status*), so the check skips it too.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from html import unescape
from pathlib import Path

from labinacan.site.research import paper_status

TERMS = "terms.toml"
PENDING = "terms_pending.txt"
EXEMPT_SECTIONS = re.compile(r"^#{2,3} (References|Reproduce)\b", re.MULTILINE)
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
LINK_TARGET = re.compile(r"\]\([^)]*\)")
FIGURE_TEXT = re.compile(r'<text[^>]*class="(?:title|sub)"[^>]*>([^<]*)</text>')


@dataclass(frozen=True)
class Term:
    """A project term: its name, the pattern that finds it, and the plain phrase to use."""

    name: str
    pattern: re.Pattern
    plain: str


def load_terms(papers: Path) -> list[Term]:
    """The project's terms, from ``terms.toml`` in ``papers``.

    Args:
        papers: The papers' folder.

    Returns:
        The terms, or [] when the project lists none.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     term = '[[term]]\\nname = "widget"\\npattern = "widgets?"\\nplain = "a part"\\n'
        ...     _ = (Path(tmp) / TERMS).write_text(term)
        ...     [t.name for t in load_terms(Path(tmp))]
        ['widget']
    """
    path = papers / TERMS
    if not path.exists():
        return []
    table = tomllib.loads(path.read_text(encoding="utf-8"))
    return [Term(t["name"], re.compile(t["pattern"]), t["plain"]) for t in table.get("term", [])]


def prose(text: str) -> str:
    """The text of a paper that a reader meets: no code, link targets, References or Reproduce.

    Args:
        text: The paper's Markdown.

    Returns:
        The text that the check reads.

    Examples:
        >>> prose("A widget.\\n\\n```\\nwidget\\n```\\n[see](widget.md)\\n## References\\nwidget")
        'A widget.\\n\\n\\n[see]\\n'
    """
    cut = EXEMPT_SECTIONS.search(text)
    text = text[: cut.start()] if cut else text
    return LINK_TARGET.sub("]", FENCE.sub("", text))


def sources(folder: Path) -> list[tuple[str, str]]:
    """Each part of a paper that the check reads, with its text.

    Args:
        folder: The paper's folder.

    Returns:
        ``(where, text)`` for the README's prose, each figure's titles and the thread.
    """
    out = [("README.md", prose((folder / "README.md").read_text(encoding="utf-8")))]
    for svg in sorted((folder / "media").glob("*.svg")):
        titles = FIGURE_TEXT.findall(svg.read_text(encoding="utf-8"))
        out.append((f"media/{svg.name}", unescape("\n".join(titles))))
    thread = folder / "thread.yaml"
    if thread.exists():
        out.append(("thread.yaml", thread.read_text(encoding="utf-8")))
    return out


def found(folder: Path, terms: list[Term]) -> list[str]:
    """Each project term in a paper, as ``<where>: '<match>' (use <plain>)``.

    Args:
        folder: The paper's folder.
        terms: The project's terms.

    Returns:
        One line for each match, in the order the parts come.
    """
    return [
        f"{where}: {match.group()!r} (use {term.plain})"
        for where, text in sources(folder)
        for term in terms
        for match in term.pattern.finditer(text)
    ]


def pending(papers: Path) -> set[str]:
    """The slugs of the preprints that wait for a rewrite, from ``terms_pending.txt``.

    Args:
        papers: The papers' folder.

    Returns:
        The slugs.
    """
    path = papers / PENDING
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def problems(papers: Path, listed: list[str]) -> list[str]:
    """The project terms in each preprint that doesn't wait, and the lines that can go.

    Args:
        papers: The papers' folder.
        listed: The slugs of the listed papers. Only preprints count: a published paper
            is frozen.

    Returns:
        Each problem as ``<slug>: <problem>``.
    """
    terms, waiting = load_terms(papers), pending(papers)
    preprints = [slug for slug in listed if _preprint(papers / slug / "README.md")]
    out = []
    for slug in preprints:
        hits = found(papers / slug, terms)
        if slug not in waiting:
            out += [f"{slug}: {hit}" for hit in hits]
        elif not hits:
            out.append(f"{slug}: no project term is left, so delete its line from {PENDING}")
    stray = sorted(waiting - set(preprints))
    return out + [f"{slug}: in {PENDING}, but not a listed preprint" for slug in stray]


def _preprint(readme: Path) -> bool:
    """Whether the paper at ``readme`` is a preprint, by its status line.

    Args:
        readme: The paper's ``README.md``.

    Returns:
        True for a preprint, False for a published paper or one with no status line.
    """
    status = paper_status(readme.read_text(encoding="utf-8"))
    return status is not None and status.preprint
