"""Hold each experiment's record slim during a site build, and load it in full on demand.

A loaded record averages about 1 MB (issue #4652). Most of it is evidence that only its
unit's pages and the Parquet export read:

- the top examples, with each token's activation;
- the other pole's top examples;
- the confound profile;
- the held-out windows that the explanation was scored on.

The ``SlimArchive`` class empties those fields as each record loads. It keeps the record's
file and claim. Then ``hand_over`` gives each experiment its file. While a block runs,
``full_records`` gives some experiments their full records, then puts the slim ones back. A
full record is made in the same steps as the build made it. It is read from the file, and
its errata's edits are applied. If the build has scrubbed it of personal data
(``pii.scrub_experiments``), it is scrubbed again.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from openinterpretability.core.errata import apply_errata, covering
from openinterpretability.core.schemas import AuditRecord

if TYPE_CHECKING:
    from openinterpretability.builder.tree import Experiment
    from openinterpretability.core.errata import Erratum


# The submission's fields that only a unit's pages and the Parquet export read, emptied.
UNIT_PAGE_ONLY = {
    "top_examples": [],
    "opposite_pole": None,
    "confounds": None,
    "input_attribution": None,
    "correlated_neurons": [],
    "patchscope": None,
    "steering": None,
    "wordnet_specificity": None,
    "vocabulary_lens": None,
}


def slim(record: AuditRecord) -> AuditRecord:
    """Return a copy of ``record`` without the evidence that only its unit's pages read."""
    sub = record.submission.model_copy(update=UNIT_PAGE_ONLY)
    verdict = record.verdict
    scores = verdict.explanation_scores if verdict else None
    if scores is not None:
        scores = scores.model_copy(update={"detection_windows": [], "fuzzing_windows": []})
        verdict = verdict.model_copy(update={"explanation_scores": scores})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


class SlimArchive:
    """Slim each record as it loads (``wiki.load_archive_paths(keep=...)``).

    It keeps, by ``id()`` of each slim record, the record's file and its claim: the hash
    of its full submission (:func:`.tree.claim_of`). A record that an erratum edits stays
    in full, since an edit may name a field that a slim record lacks.
    """

    def __init__(self, errata: Sequence[Erratum]):
        self.errata = errata
        self.paths: dict[int, Path] = {}
        self.claims: dict[int, str] = {}

    def __call__(self, path: Path, record: AuditRecord) -> AuditRecord:
        """Return what to hold of ``record``, loaded from ``path``: a slim copy, if it can."""
        from openinterpretability.builder.tree import claim_of

        if record.verdict is None or any(e.edits for e in covering(self.errata, record)):
            return record
        held = slim(record)
        self.paths[id(held)], self.claims[id(held)] = path, claim_of(record)
        return held


def hand_over(records: list, experiments: Sequence[Experiment], paths: dict | None) -> None:
    """Give each experiment of a slim record its file, so it can be loaded in full again.

    The caller hands over ``records``, and the list is emptied. Without ``paths`` (tests),
    nothing changes.
    """
    if not paths:
        return
    for e in experiments:
        e.source = paths.get(e.loaded)
    records.clear()


def full_record(e: Experiment) -> AuditRecord:
    """Return ``e``'s record in full, as the build holds it, or the record itself if not slim."""
    if e.source is None:
        return e.record
    from openinterpretability.builder.pii import scrub_record

    record, _ = apply_errata(AuditRecord.model_validate_json(e.source.read_bytes()), e.errata)
    return scrub_record(record) if e.scrubbed else record


@contextmanager
def full_records(experiments: Sequence[Experiment]) -> Iterator[None]:
    """Give ``experiments`` their full records while the block runs, then the slim ones."""
    held = [(e, e.record) for e in experiments if e.source is not None]
    for e, _ in held:
        e.record = full_record(e)
    try:
        yield
    finally:
        for e, record in held:
            e.record = record
