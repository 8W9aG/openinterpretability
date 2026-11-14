"""Schemas for where a scan's corpus came from.

:class:`CorpusSource` is the sidecar of the dataset sample that ``open-interpretability corpus``
wrote. :class:`CorpusSample` is the token-bounded part of it that an audit read. Both
are built by ``openinterp.corpus``.
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel


class CorpusSource(StrictModel):
    """Where a scan corpus came from.

    ``open-interpretability corpus`` writes it next to the corpus.

    Enough to rebuild the same documents: the dataset at an exact commit, the files and
    column read, the seed and sizes. ``license`` is the dataset card's licence, so pages
    that show excerpts can attribute them.
    """

    dataset: str = Field(min_length=1, examples=["HuggingFaceFW/fineweb-edu"])
    revision: str | None = Field(default=None, description="Dataset commit hash.")
    files: str = Field(
        description="Glob of the Parquet files sampled, within the dataset; several, "
        "comma-separated, each gave an equal share of the documents."
    )
    text_column: str = Field(
        default="text",
        description="Column of the text; several, comma-separated, joined into one document "
        "with a blank line between them.",
    )
    documents: int = Field(ge=0)
    seed: int
    max_chars: int = Field(gt=0)
    docs_per_row_group: int | None = Field(default=None, gt=0, description="None: a blend.")
    layout: str | None = Field(
        default=None,
        description="How rows were joined into documents (e.g. 'wikitext'); None: one per row.",
    )
    first: bool = False  # the first documents in file order, not a seeded sample (no strip)
    ascii: bool = False  # non-ASCII \u-escaped in the JSONL, as in the registry's corpus
    lang_pattern: str | None = Field(
        default=None,
        description="Regular expression whose group, matched in each file glob, gave its "
        "documents' 'lang' field; None: documents carry no language.",
    )
    lang_detector: str | None = Field(
        default=None,
        description="The detector that gave each document's 'lang' field "
        "(openinterp.langid.DETECTOR), when the dataset doesn't say; documents it was "
        "unsure of have none. None: no detected languages.",
    )
    code_language_column: str | None = Field(
        default=None,
        description="The dataset column that gave each document's 'code_language' field; "
        "None: documents carry no programming language.",
    )
    code_detector: str | None = Field(
        default=None,
        description="The detector that gave each document's 'code_language' field "
        "(openinterp.codeid.detector()), when the dataset doesn't say; documents it was "
        "unsure of, and prose, have none. None: no detected programming languages.",
    )
    chat_template: str | None = Field(
        default=None,
        description="The model whose tokenizer's chat template wrote each conversation in "
        "the text column as a document; None: the column's text, as it is.",
    )
    chat_template_revision: str | None = Field(
        default=None, description="Commit of the chat template's model."
    )
    license: str | None = None
    sha256: str = Field(description="SHA-256 of the corpus file this describes.")
    blend: str | None = Field(
        default=None,
        description="The versioned blend of presets this corpus is (openinterp.blends.BLENDS), "
        "e.g. 'mixed-v1'; its documents are its components' in order, each recording its "
        "preset as 'source'. None: one dataset.",
    )
    components: list[CorpusSource] | None = Field(
        default=None,
        description="A blend's samples, one per preset, each as `open-interpretability corpus` "
        "writes that preset alone with the same seed and its share of the documents.",
    )

    def __str__(self) -> str:
        how = f"{self.documents} docs, seed {self.seed}"
        if self.blend:
            return f"{self.blend} ({len(self.components or [])} presets, {how})"
        at = f"@{self.revision[:8]}" if self.revision else ""
        how = f"first {self.documents} docs" if self.first else how
        if self.chat_template:
            how += f", {self.chat_template}'s chat template"
        return f"{self.dataset}{at} ({self.files}, {how})"


class CorpusSample(StrictModel):
    """The seeded, token-bounded sample of the corpus a scan read (protocol 0.7).

    The sampler visits documents in ``random.Random(seed)`` order. It takes each document
    whole if it fits in what is left of ``max_tokens``, and always takes the first. Then
    it puts the documents back in corpus order. The model's tokenizer counts the tokens,
    without special tokens.
    Re-drawing with the same corpus, tokenizer, seed and budget gives the same
    documents, which ``sha256`` pins.
    """

    seed: int
    max_tokens: int = Field(gt=0, description="Token budget the sample was drawn to.")
    documents: int = Field(gt=0)
    corpus_documents: int = Field(gt=0, description="Documents in the corpus it was drawn from.")
    tokens: int = Field(ge=0, description="Tokenizer tokens in the sampled documents.")
    sha256: str = Field(
        description='SHA-256 of the sampled documents as JSONL ({"text": ...} per line), '
        "the bytes `open-interpretability corpus` would write for them."
    )

    def __str__(self) -> str:
        return (
            f"{self.documents} of {self.corpus_documents} docs, {self.tokens:,} tokens "
            f"(≤ {self.max_tokens:,}, seed {self.seed})"
        )
