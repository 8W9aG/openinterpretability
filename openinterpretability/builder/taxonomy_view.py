"""The About page's view of the taxonomy (:mod:`openinterpretability.core.taxonomy`, #3703)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from markupsafe import Markup, escape

from openinterpretability.core.taxonomy import TAXONOMY, Concept

_CODE = re.compile(r"`([^`]+)`")
_ISSUE = re.compile(r"#(\d+)")
_TOKEN = re.compile(r"(`[^`]+`|#\d+)")


@dataclass(frozen=True)
class ConceptView:
    """One row of the About page's taxonomy table."""

    concept: Concept
    sister: Markup  # the sister text, with code and issue links


def inline(text: str, repo_url: str) -> Markup:
    """``text`` as HTML: backticked spans as code, and ``#N`` as a link to issue N.

    Every piece of ``text`` is escaped, so no part of it can add markup of its own.
    """
    code = Markup("<code>{}</code>")
    link = Markup('<a class="underline underline-offset-2" href="{}/issues/{}">#{}</a>')
    parts = []
    for piece in _TOKEN.split(text):
        if _CODE.fullmatch(piece):
            parts.append(code.format(piece[1:-1]))
        elif _ISSUE.fullmatch(piece):
            parts.append(link.format(repo_url, piece[1:], piece[1:]))
        else:
            parts.append(escape(piece))
    return Markup("").join(parts)


def taxonomy_rows(repo_url: str) -> list[ConceptView]:
    """Each concept of the taxonomy, in its order, for the About page."""
    return [ConceptView(c, inline(c.sister, repo_url)) for c in TAXONOMY]
