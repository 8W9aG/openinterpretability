"""A table of contents for long pages, built from each page's own headings (#3756).

Every page that the site renders goes through :func:`add_contents`. The top-level sections
are the headings at the highest level below ``<h1>`` that the page uses inside ``<main>``:
``<h2>`` on papers and most pages, ``<h3>`` on experiment pages. A heading with no ``id``
gets one from its text, so an anchor stays the same while the heading's text does.

A page with at least :data:`MIN_SECTIONS` top-level sections gets two lists of links. A
collapsed one sits under the ``<h1>``. A fixed one sits in the right margin, shown only
on screens wide enough to have a margin. Only one of the two is displayed at a time.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from markupsafe import Markup

MIN_SECTIONS = 5  # fewer top-level sections than this, and a page has no table of contents

_MAIN = re.compile(r"(<main\b[^>]*>)(.*?)(</main>)", re.S)
_HEADING = re.compile(r"<h([23])\b([^>]*)>(.*?)</h\1>", re.S)
_ID = re.compile(r'\bid="([^"]+)"')
_TAG = re.compile(r"<[^>]+>")
_H1_END = "</h1>"


@dataclass(frozen=True)
class Section:
    """One top-level section: its anchor and its heading's text."""

    anchor: str
    title: str


def anchor(title: str, taken: set[str]) -> str:
    """A stable id for a section, from its title: lower case, hyphens, and unique on the page.

    The id joins ``taken``. A repeated title gets ``-2``, ``-3`` and so on.
    """
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    found, n = base, 2
    while found in taken:
        found, n = f"{base}-{n}", n + 1
    taken.add(found)
    return found


def _text(inner: str) -> str:
    """A heading's plain text: its tags removed and its entities decoded."""
    return " ".join(html.unescape(_TAG.sub("", inner)).split())


def _sections(main: str) -> tuple[str, list[Section]]:
    """``main`` with an id on each top-level heading, and the top-level sections."""
    found = _HEADING.findall(main)
    if not found:
        return main, []
    top = min(level for level, _, _ in found)
    taken = set(_ID.findall(main))
    sections: list[Section] = []

    def mark(match: re.Match) -> str:
        level, attrs, inner = match.groups()
        if level != top:
            return match.group(0)
        title = _text(inner)
        given = _ID.search(attrs)
        name = given.group(1) if given else anchor(title, taken)
        sections.append(Section(name, title))
        attrs = attrs if given else f'{attrs} id="{name}"'
        return f"<h{level}{attrs}>{inner}</h{level}>"

    return _HEADING.sub(mark, main), sections


def _links(sections: list[Section]) -> Markup:
    item = Markup('<li><a class="hover:underline" href="#{}">{}</a></li>')
    return Markup("").join(item.format(s.anchor, s.title) for s in sections)


def contents_nav(sections: list[Section]) -> Markup:
    """The two lists of links: collapsed under the ``<h1>``, and fixed in the margin."""
    links = _links(sections)
    inline = Markup(
        '<nav aria-label="Contents" class="my-4 text-sm 2xl:hidden" data-pagefind-ignore>'
        '<details class="rounded-lg border border-slate-200 px-3 py-2 dark:border-slate-800">'
        '<summary class="cursor-pointer font-semibold">Contents</summary>'
        '<ol class="mt-2 list-decimal space-y-1 pl-5 text-slate-600 dark:text-slate-400">{}</ol>'
        "</details></nav>"
    ).format(links)
    side = Markup(
        '<nav aria-label="Contents" class="fixed right-4 top-24 hidden max-h-[75vh] w-48 '
        'overflow-y-auto text-xs 2xl:block" data-pagefind-ignore>'
        '<p class="mb-2 font-semibold text-slate-700 dark:text-slate-300">Contents</p>'
        '<ol class="space-y-1.5 text-slate-500 dark:text-slate-400">{}</ol></nav>'
    ).format(links)
    return inline + side


def add_contents(page: str) -> str:
    """``page`` with ids on its top-level headings, and a table of contents if it is long."""
    found = _MAIN.search(page)
    if found is None:
        return page
    main, sections = _sections(found.group(2))
    if len(sections) >= MIN_SECTIONS and _H1_END in main:
        main = main.replace(_H1_END, _H1_END + str(contents_nav(sections)), 1)
    return page[: found.start(2)] + main + page[found.end(2) :]
