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
    """The length of ``text`` as X counts it, without links.

    Args:
        text: The text to count.

    Returns:
        The number of characters as X counts them.

    Examples:
        >>> weight("neuron")
        6
        >>> weight("神经元")  # each CJK character counts as two
        6
    """
    return sum(1 if any(a <= ord(c) <= b for a, b in LIGHT) else 2 for c in text)


def fit(text: str, budget: int) -> str:
    """``text`` cut at a word boundary, with an ellipsis, so its weight is at most ``budget``.

    Args:
        text: The text to cut.
        budget: The largest weight allowed.

    Returns:
        ``text`` itself if it fits, else its first words and an ellipsis, or "" if no word fits.

    Examples:
        >>> fit("Fires on the closing bracket", 20)
        'Fires on the…'
        >>> fit("Short", 20)
        'Short'
    """
    if weight(text) <= budget:
        return text
    words = text.split()
    while words and weight(" ".join(words) + ELLIPSIS) > budget:
        words.pop()
    return " ".join(words).rstrip(",;:.") + ELLIPSIS if words else ""


def first_sentence(text: str) -> str:
    """The first sentence of ``text``: up to its first full stop, question or exclamation.

    Args:
        text: The text, such as a paper's abstract.

    Returns:
        The first sentence, or the whole stripped text if it has no end mark.

    Examples:
        >>> first_sentence("It reads brackets. More follows.")
        'It reads brackets.'
    """
    found = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return found.group(1) if found else text.strip()


def compose(head: str, body: str, link: str, separator: str = " ") -> str:
    r"""``head`` and ``body``, cut to fit with ``link`` on its own line, and screened.

    Args:
        head: The post's first words, kept whole.
        body: The text after them, cut to fit.
        link: The link, on its own last line.
        separator: What goes between ``head`` and ``body``.

    Returns:
        The post: the text on one or more lines, then the link on the last line.

    Examples:
        >>> compose("L3/N7+ VERIFIED:", "Fires on dates.", "https://x.org/u")
        'L3/N7+ VERIFIED: Fires on dates.\nhttps://x.org/u'
    """
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
    r"""The post for a VERIFIED experiment, from its ``feed.xml`` entry: unit, verdict and link.

    The post carries no text from the record. Any contributor can write an experiment
    record, and it can merge with no person reading it. So its explanation never goes out
    under the project's account. The link leads to the unit's page, which shows it.

    Args:
        entry: The experiment's feed entry.

    Returns:
        The post's text, which fits :data:`LIMIT`.

    Examples:
        >>> experiment_post(Entry("u", "gpt2/L1/N5-positive VERIFIED", "https://x.org/u",
        ...                       "Fires on dates. @someone https://elsewhere.example"))
        'gpt2/L1/N5-positive VERIFIED\nhttps://x.org/u'
    """
    slug = entry.title.removesuffix(" VERIFIED")
    return compose(f"{slug} VERIFIED", "", entry.link)


def paper_post(entry: Entry) -> str:
    r"""The post for a published paper, from its ``research/published.xml`` entry.

    Args:
        entry: The paper's feed entry.

    Returns:
        The post's text, which fits :data:`LIMIT`.

    Examples:
        >>> paper = Entry("p", "A bracket neuron", "https://x.org/p", "It reads brackets. More.")
        >>> paper_post(paper)
        'A bracket neuron\n\nIt reads brackets.\nhttps://x.org/p'
    """
    return compose(entry.title, first_sentence(entry.summary), entry.link, separator="\n\n")


def digest_post(count: int, link: str) -> str:
    r"""One post for many new VERIFIED units at once.

    Args:
        count: How many new VERIFIED units.
        link: The site's link.

    Returns:
        The post's text.

    Examples:
        >>> digest_post(5, "https://x.org/")
        '5 new VERIFIED units in the registry.\nhttps://x.org/'
    """
    return f"{count} new VERIFIED units in the registry.\n{link}"


def length(post: str) -> int:
    r"""The weighted length of a whole post, each link counted as :data:`LINK`.

    Args:
        post: The whole post, links included.

    Returns:
        The weight of the text outside links, plus :data:`LINK` for each link.

    Examples:
        >>> length("Fires on dates.\nhttps://example.org/a/very/long/path")
        39
    """
    links = re.findall(r"https?://\S+", post)
    rest = re.sub(r"https?://\S+", "", post)
    return weight(rest) + LINK * len(links)
