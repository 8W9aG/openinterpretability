"""The text of a post: what it says, cut to fit, and screened as the site is (issue #4463).

- A VERIFIED experiment: ``<unit slug> VERIFIED: <explanation>``, then its link.
- A published paper: its title, then the first sentence of its abstract, then its link.

X counts a post's length its own way. Every link counts as :data:`LINK` characters. Most
characters outside the Latin, Greek and Cyrillic ranges (CJK, emoji) count as two.
The explanation or the sentence is cut at a word boundary, with an ellipsis, so the post
fits :data:`LIMIT`. The text passes the site's search for personal data and taken-down
passages (``builder.pii.redact_text``) before it is posted, as every page does.
"""

from __future__ import annotations

import re

from openinterpretability.social.feed import Entry

LIMIT = 280  # X's maximum weighted length of a post
LINK = 23  # what X counts for any link
ELLIPSIS = "…"
# Code point ranges that X counts as one character each; every other one counts as two.
LIGHT = ((0x0000, 0x10FF), (0x2000, 0x200D), (0x2010, 0x201F), (0x2032, 0x2037))


def weight(text: str) -> int:
    """The length of ``text`` as X counts it, without links."""
    return sum(1 if any(a <= ord(c) <= b for a, b in LIGHT) else 2 for c in text)


def fit(text: str, budget: int) -> str:
    """``text`` cut at a word boundary, with an ellipsis, so its weight is at most ``budget``."""
    if weight(text) <= budget:
        return text
    words = text.split()
    while words and weight(" ".join(words) + ELLIPSIS) > budget:
        words.pop()
    return " ".join(words).rstrip(",;:.") + ELLIPSIS if words else ""


def first_sentence(text: str) -> str:
    """The first sentence of ``text``: up to its first full stop, question or exclamation."""
    found = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return found.group(1) if found else text.strip()


def compose(head: str, body: str, link: str, separator: str = " ") -> str:
    """``head`` and ``body``, cut to fit with ``link`` on its own line, and screened."""
    from openinterpretability.builder.pii import redact_text

    head, body = redact_text(head), redact_text(body)
    budget = LIMIT - LINK - 1 - weight(head) - weight(separator)
    text = (
        f"{head}{separator}{fit(body, budget)}"
        if budget > 0 and body
        else fit(head, LIMIT - LINK - 1)
    )
    return f"{text}\n{link}"


def experiment_post(entry: Entry) -> str:
    """The post for a VERIFIED experiment, from its ``feed.xml`` entry."""
    slug = entry.title.removesuffix(" VERIFIED")
    return compose(f"{slug} VERIFIED:", entry.summary, entry.link)


def paper_post(entry: Entry) -> str:
    """The post for a published paper, from its ``research/published.xml`` entry."""
    return compose(entry.title, first_sentence(entry.summary), entry.link, separator="\n\n")


def digest_post(count: int, link: str) -> str:
    """One post for many new VERIFIED units at once."""
    return f"{count} new VERIFIED units in the registry.\n{link}"


def length(post: str) -> int:
    """The weighted length of a whole post, each link counted as :data:`LINK`."""
    links = re.findall(r"https?://\S+", post)
    rest = re.sub(r"https?://\S+", "", post)
    return weight(rest) + LINK * len(links)
