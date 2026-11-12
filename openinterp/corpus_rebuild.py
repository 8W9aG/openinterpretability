"""Rebuild the corpus that a record scanned (`openinterp corpus --same-as`).

A follow-up review draws held-out windows from the scan's corpus (`openinterp score`).
Corpora are git-ignored, and a scan can read a corpus that only one experiment used. So
the record's source is what brings the same bytes back.
"""

from __future__ import annotations

from pathlib import Path

from openinterp.blends import build_blend
from openinterp.core.schemas import CorpusSource
from openinterp.corpus import build_from_hub, load_sources, source_path

# Records that scanned this corpus before they recorded a source (README, "A scan corpus").
COMMITTED_SIDECARS = (Path("corpus.source.json"),)


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
