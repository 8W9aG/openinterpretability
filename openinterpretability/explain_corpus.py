"""The corpus whose means are ``explain``'s baseline, when no ``--corpus`` is given (#5829).

A unit's mean on a corpus is the value that mean ablation puts in its place, so the
attribution depends on the corpus. Every explanation of a model must use the same one, or
``explain`` can't reuse an earlier explanation (``explain_reuse``). :data:`EXPLAIN_CORPUS`
is that corpus: the first 500 documents of pile-10k at a pinned commit, each cut to 3,000
characters. It is the corpus that almost every scan in the registry read (SHA-256
``ded48cfc…``, the committed ``corpus.source.json``).

``corpus.jsonl`` is git-ignored, and a pip install has no copy of it. So
:func:`default_corpus` builds the corpus once into the cache, and later runs reuse it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from openinterpretability.core.corpora import CorpusSource

# The registry's scan corpus. tests/test_explain_corpus.py checks that it equals the
# committed corpus.source.json, so the two never drift.
EXPLAIN_CORPUS = CorpusSource(
    dataset="NeelNanda/pile-10k",
    revision="127bfedcd5047750df5ccf3a12979a47bfa0bafa",
    files="data/*.parquet",
    text_column="text",
    documents=500,
    seed=0,
    max_chars=3000,
    docs_per_row_group=20,
    first=True,
    ascii=True,
    license="bigscience-bloom-rail-1.0",
    sha256="ded48cfc4898c00bd663c4e26dd02baddc48f54f53469988097030686dd2a2ad",
)
LOCAL = Path("corpus.jsonl")  # the registry's corpus, where a clone has built it


def default_corpus(
    echo: Callable[[str], None] = print,
    source: CorpusSource = EXPLAIN_CORPUS,
    local: Path = LOCAL,
) -> Path:
    """The path of ``source``'s corpus: ``local`` if it holds those bytes, else the cache.

    The first run without a copy builds the corpus into the rebuilt-corpora cache
    (:func:`.corpus_rebuild.cached`), and says so first. Later runs reuse it.

    Args:
        echo: Where the line about the build goes.
        source: The corpus to use.
        local: A copy that may already hold it.

    Returns:
        A corpus file with ``source``'s SHA-256. When the cache is off, or the build gives
        other bytes, :func:`.corpus_rebuild.cached` raises ``ValueError``.

    Examples:
        >>> import hashlib, tempfile
        >>> local = Path(tempfile.mkdtemp()) / "corpus.jsonl"
        >>> _ = local.write_text('{"text": "one"}\\n')
        >>> digest = hashlib.sha256(local.read_bytes()).hexdigest()
        >>> mine = EXPLAIN_CORPUS.model_copy(update={"sha256": digest})
        >>> default_corpus(source=mine, local=local) == local
        True
    """
    import hashlib

    if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == source.sha256:
        return local
    return _found(source) or _built(source, echo)


def _found(source: CorpusSource) -> Path | None:
    """The cached copy of ``source``'s corpus, or None if there is none yet or no cache.

    Args:
        source: The corpus to find.

    Returns:
        The cached corpus file, or None.
    """
    from openinterpretability.corpus_rebuild import cache_path

    out = cache_path(source)
    return out if out and out.is_file() else None


def _built(source: CorpusSource, echo: Callable[[str], None]) -> Path:
    """Build ``source``'s corpus into the cache, after saying so.

    Args:
        source: The corpus to build.
        echo: Where the line about the build goes.

    Returns:
        The built corpus file.
    """
    from openinterpretability.corpus_rebuild import cached

    echo(f"Building the explain corpus ({source.dataset}, {source.documents} documents), "
         "once. Later runs reuse it.")  # fmt: skip
    return cached(source, "the explain corpus can't be built. Pass --corpus with a corpus file")
