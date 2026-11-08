"""How a dataset's rows become documents, for the layouts of ``openinterp corpus``."""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import pairwise

# WikiText's top-level heading row (" = Title = \n") starts an article; sections are
# " = = Section = = \n". Its text is tokenised: "1 @,@ 000", "well @-@ known", "lobster ,".
_WIKITEXT_TITLE = re.compile(r" = [^=].* = \n?")
_WIKITEXT_HEADING = re.compile(r"(?:= )+(.*?)(?: =)+")
_WIKITEXT_DETOKENIZE = [
    (re.compile(r" @(.)@ "), r"\1"),  # number separators and hyphens
    (re.compile(r" ([.,;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$]) "), r"\1"),
    (re.compile(r" (n't|'s|'re|'ve|'m|'ll|'d)\b"), r"\1"),
    (re.compile(r" ' s\b"), "'s"),  # possessive after punctuation: "Inc. ' s"
    (re.compile(r"s ' "), "s' "),  # plural possessive
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),
]


def wikitext_articles(rows: Sequence[str | None]) -> list[str]:
    """Return the WikiText articles that start and end within ``rows``, detokenised.

    Headings become plain lines, and blank lines separate paragraphs. An article cut by
    the start or end of ``rows`` (a row group's edge) is dropped, not truncated.
    """
    starts = [i for i, row in enumerate(rows) if _WIKITEXT_TITLE.fullmatch(row or "")]
    return [_wikitext_article(rows[a:b]) for a, b in pairwise(starts)]


def _wikitext_article(rows: Sequence[str | None]) -> str:
    paragraphs = []
    for row in rows:
        text = (row or "").strip()
        if heading := _WIKITEXT_HEADING.fullmatch(text):
            text = heading.group(1)
        for pattern, repl in _WIKITEXT_DETOKENIZE:
            text = pattern.sub(repl, text)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


# Row layouts other than one document per row: rows of one row group -> its documents.
LAYOUTS = {"wikitext": wikitext_articles}


def chat_formatter(model: str, revision: str | None = None):
    """Return a function that writes a conversation with ``model``'s chat template.

    Also returns the model commit that the template comes from. A conversation is a list
    of ``{"role": ..., "content": ...}`` messages. The text keeps the template's special
    tokens as text (e.g. ``<|im_start|>``), which the tokenizer reads back as the same
    tokens. Raises ``ValueError`` if the model's tokenizer has no chat template.
    """
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().model_info(model, revision=revision).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError(f"{model}'s tokenizer has no chat template")

    def render(messages) -> str:
        return tokenizer.apply_chat_template(list(messages or []), tokenize=False)

    return render, revision
