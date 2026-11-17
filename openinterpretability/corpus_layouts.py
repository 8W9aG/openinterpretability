"""How a dataset's rows become documents, for the layouts of ``open-interpretability corpus``."""

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


# The parts of a book that are left out: its first 10% and its last 5%. They hold its front
# matter (producer credits, title page, contents, preface) and back matter (notes, index).
BOOK_FRONT = 0.10
BOOK_BACK = 0.05
PASSAGE_CHARS = (
    3000  # a passage's length, at most: the default of `open-interpretability corpus --max-chars`
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")


def book_passages(rows: Sequence[str | None]) -> list[str]:
    """Return passages from the body of each book in ``rows``, in order.

    A book is one row, often hundreds of thousands of characters. Cut to its start, it
    would give only front matter. So each book loses its first :data:`BOOK_FRONT` and last
    :data:`BOOK_BACK`. The paragraphs at those cuts can be partial, so they are dropped
    too. The paragraphs left join into passages of at most :data:`PASSAGE_CHARS`, which
    always start and end at a paragraph break. A longer paragraph is a passage alone. The
    books' lines are wrapped at about 70 characters, so a line break within a paragraph
    becomes a space. Verse loses its line breaks too.
    """
    passages = []
    for book in rows:
        text = book or ""
        body = text[int(len(text) * BOOK_FRONT) : len(text) - int(len(text) * BOOK_BACK)]
        paragraphs = [_LINE_BREAK.sub(" ", p).strip() for p in _PARAGRAPH_BREAK.split(body)]
        passage: list[str] = []
        for paragraph in filter(None, paragraphs[1:-1]):
            if passage and len("\n\n".join([*passage, paragraph])) > PASSAGE_CHARS:
                passages.append("\n\n".join(passage))
                passage = []
            passage.append(paragraph)
        if passage:
            passages.append("\n\n".join(passage))
    return passages


# Row layouts other than one document per row: rows of one row group -> its documents.
LAYOUTS = {"wikitext": wikitext_articles, "book_passages": book_passages}


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
