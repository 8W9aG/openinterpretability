"""Schemas for where a scan's corpus came from: the dataset sample ``openinterp corpus``
wrote (:class:`CorpusSource`, its sidecar) and the token-bounded sample of it an audit
read (:class:`CorpusSample`). Built by ``openinterp.corpus``."""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel


class CorpusSource(StrictModel):
    """Where a scan corpus came from, written by ``openinterp corpus`` next to the corpus.

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
    text_column: str = "text"
    documents: int = Field(ge=0)
    seed: int
    max_chars: int = Field(gt=0)
    docs_per_row_group: int = Field(gt=0)
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
    license: str | None = None
    sha256: str = Field(description="SHA-256 of the corpus file this describes.")

    def __str__(self) -> str:
        at = f"@{self.revision[:8]}" if self.revision else ""
        how = f"{self.documents} docs, seed {self.seed}"
        how = f"first {self.documents} docs" if self.first else how
        return f"{self.dataset}{at} ({self.files}, {how})"


class CorpusSample(StrictModel):
    """The seeded, token-bounded sample of the corpus a scan read (protocol 0.7).

    Documents are visited in ``random.Random(seed)`` order, and each is taken whole if it
    fits in what is left of ``max_tokens`` (the first always is), then kept in corpus
    order. Tokens are counted by the model's tokenizer, without special tokens.
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
        "the bytes `openinterp corpus` would write for them."
    )

    def __str__(self) -> str:
        return (
            f"{self.documents} of {self.corpus_documents} docs, {self.tokens:,} tokens "
            f"(≤ {self.max_tokens:,}, seed {self.seed})"
        )
