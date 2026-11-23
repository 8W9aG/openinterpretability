"""Rebuild the corpus that a record scanned (`open-interpretability corpus --same-as`).

A follow-up review draws held-out windows from the scan's corpus (`open-interpretability score`).
Corpora are git-ignored, and a scan can read a corpus that only one experiment used. So
the record's source is what brings the same bytes back.
"""

from __future__ import annotations

import re
from pathlib import Path

from openinterpretability.blends import build_blend
from openinterpretability.core.schemas import COMMITTED_SOURCES, CorpusSource
from openinterpretability.corpus import build_from_hub, load_sources, source_path

COMMITTED_SIDECARS = tuple(Path(name) for name in COMMITTED_SOURCES)
# A SHA-256 as the cache names its files: 64 lowercase hex digits, so no path can come of it.
SHA256 = re.compile(r"[0-9a-f]{64}")


def scanned_source(submission) -> CorpusSource:
    """Return the source of the corpus that ``submission`` scanned.

    A record without its own source can match a committed sidecar by SHA-256. Raises
    ``ValueError`` if neither says how to rebuild the corpus.
    """
    provenance = submission.provenance
    if provenance is None:
        raise ValueError("the record has no provenance, so it names no corpus")
    source = provenance.corpus_source or load_sources(COMMITTED_SIDECARS).get(
        provenance.corpus_sha256
    )
    if source is None:
        raise ValueError(
            f"the record doesn't say how to rebuild its corpus ({provenance.corpus}, "
            f"SHA-256 {provenance.corpus_sha256})"
        )
    return source


def rebuild(source: CorpusSource, out: Path) -> CorpusSource:
    """Write the corpus that ``source`` describes to ``out``, at its pinned dataset commits.

    Raises ``ValueError``, and leaves no corpus, if the bytes differ. A scan's held-out
    windows must come from its own text, so a near match is not acceptable.
    """
    built = _build(source, out)
    if built.sha256 != source.sha256:
        out.unlink()
        source_path(out).unlink()
        raise ValueError(
            f"the rebuilt corpus has SHA-256 {built.sha256}, but the record scanned "
            f"{source.sha256}. Its dataset or preset may have changed."
        )
    return built


def cache_path(source: CorpusSource) -> Path | None:
    """Where the rebuilt-corpora cache keeps ``source``'s corpus, or None when it is off.

    Args:
        source: The corpus.

    Returns:
        ``<cache>/<sha256>.jsonl``. The cache is ``$OPENINTERPRETABILITY_REBUILT_CORPORA``,
        else ``rebuilt-corpora`` in the cache folder.

    Raises:
        ValueError: ``source.sha256`` is not 64 lowercase hex digits. A record gives the
            source, so the name must not be able to point outside the cache.

    Examples:
        >>> import os
        >>> os.environ["OPENINTERPRETABILITY_REBUILT_CORPORA"] = "/cache"
        >>> from openinterpretability.explain_corpus import EXPLAIN_CORPUS
        >>> cache_path(EXPLAIN_CORPUS).name[:8], str(cache_path(EXPLAIN_CORPUS).parent)
        ('ded48cfc', '/cache')
        >>> del os.environ["OPENINTERPRETABILITY_REBUILT_CORPORA"]
    """
    from openinterpretability.core.cache_dirs import cache_root

    if not SHA256.fullmatch(source.sha256):
        raise ValueError(f"not a SHA-256, so the corpus can't be cached: {source.sha256!r}")
    root = cache_root("OPENINTERPRETABILITY_REBUILT_CORPORA", "rebuilt-corpora")
    return root / f"{source.sha256}.jsonl" if root else None


def cached(source: CorpusSource, reason: str = "it can't be built") -> Path:
    """The corpus that ``source`` describes, built once into the cache (:func:`cache_path`).

    The corpus's source sidecar is written beside it.

    Args:
        source: The corpus to build or find.
        reason: What fails without the cache, for the error.

    Returns:
        The corpus file's path.

    Raises:
        ValueError: The cache is off, or the build gives other bytes (:func:`rebuild`).

    Examples:
        >>> import os
        >>> os.environ["OPENINTERPRETABILITY_REBUILT_CORPORA"] = "off"
        >>> from openinterpretability.explain_corpus import EXPLAIN_CORPUS
        >>> cached(EXPLAIN_CORPUS, "the example needs it")
        Traceback (most recent call last):
        ValueError: the corpus cache is off, so the example needs it
        >>> del os.environ["OPENINTERPRETABILITY_REBUILT_CORPORA"]
    """
    out = cache_path(source)
    if out is None:
        raise ValueError(f"the corpus cache is off, so {reason}")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        rebuild(source, out)
    return out


def _build(source: CorpusSource, out: Path) -> CorpusSource:
    if source.blend:
        return build_blend(
            source.blend,
            out,
            documents=source.documents,
            seed=source.seed,
            max_chars=source.max_chars,
            ascii=source.ascii,
            revisions=_revisions(source.components or []),
        )
    return build_from_hub(
        source.dataset,
        out,
        files=source.files,
        text_column=source.text_column,
        revision=source.revision,
        documents=source.documents,
        seed=source.seed,
        max_chars=source.max_chars,
        docs_per_row_group=source.docs_per_row_group,
        layout=source.layout,
        first=source.first,
        ascii=source.ascii,
        lang_pattern=source.lang_pattern,
        detect_lang=source.lang_detector is not None,
        code_language_column=source.code_language_column,
        detect_code=source.code_detector is not None,
        chat_template=source.chat_template,
        chat_template_revision=source.chat_template_revision,
    )


def _revisions(components: list[CorpusSource]) -> dict[str, str | None]:
    """A blend's pinned commits: each dataset's, and each chat template model's."""
    revisions = {part.dataset: part.revision for part in components}
    templates = {p.chat_template: p.chat_template_revision for p in components if p.chat_template}
    return revisions | templates
