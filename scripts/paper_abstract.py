"""A paper's abstract: its text and its word count, against the preprint's cap (#6929).

``scripts/papers.py`` reads both. Its ``check`` fails a preprint whose abstract has more
than ``ABSTRACT_WORDS`` words (write-paper, Structure).
"""

from __future__ import annotations

import re


def abstract(text: str) -> str:
    r"""The paper's abstract as written: the first paragraph after its title, maths kept.

    Args:
        text: The paper's README.

    Returns:
        The paragraph, on one line.

    Examples:
        >>> abstract("# T\n\nIt fits $R^2$ = 0.5\nwell.\n\n**Status:** x")
        'It fits $R^2$ = 0.5 well.'
    """
    body = text.split("\n\n", 1)[1].lstrip("\n")
    return " ".join(body.split("\n\n", 1)[0].split())


ABSTRACT_WORDS = 220  # the abstract's word cap for a preprint (write-paper, Structure, #6929)


def abstract_words(text: str) -> int:
    r"""The words of the paper's abstract, with each ``$…$`` formula counted as one.

    Args:
        text: The paper's README.

    Returns:
        The count.

    Examples:
        >>> abstract_words("# T\n\nIt fits $R^2 = 0.5$ well.\n\n**Status:** x")
        4
    """
    return len(re.sub(r"\$[^$]*\$", "formula", abstract(text)).split())
