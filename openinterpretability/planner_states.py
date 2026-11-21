"""What the planner knows of each studied unit: its latest counted record and its history.

:func:`load_states` reads the archive with the errata of each record applied
(:mod:`openinterpretability.core.errata`, issue #3271). So the planner sees the standings of the
site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openinterpretability.core.agent_names import canonical_agent
from openinterpretability.core.schemas import AuditRecord


@dataclass
class NeuronState:
    """What the registry holds on one unit: its latest record and how much it has been studied."""

    slug: str
    record: AuditRecord
    record_path: Path
    experiments: int = 0
    reviewers: set[str] = field(default_factory=set)
    independent_reviews: int = 0  # experiments reviewed by a model other than the proposer's
    last: datetime | None = None
    contested: bool = False  # its standing is CONTESTED (#3212, #4654)
    # A newer hypothesis failed after an earlier VERIFIED one, so a new hypothesis settles
    # it, not one more review (#4654).
    contested_hypothesis: bool = False
    # The standing (``builder.standing``): its status and the record it rests on. These can
    # differ from ``record``, the newest: a newer NEEDS_REVISION leaves a VERIFIED
    # hypothesis standing (#4900). ``explain`` reads these, the follow-ups read ``record``.
    status: str = ""
    standing_record: AuditRecord | None = None

    @property
    def standing(self) -> tuple[str, AuditRecord]:
        """The standing's status and record, or the newest record's when none is derived."""
        return (self.status or self.record.verdict.status), (self.standing_record or self.record)


def load_states(wiki_dir: Path) -> dict[str, NeuronState]:
    """One state per neuron pole, derived from all of its archived experiments.

    Each record has the edits of its errata applied (:mod:`openinterpretability.core.errata`).
    Follow-ups are planned from the most recent experiment that counts toward the status.
    An experiment that an erratum sets aside doesn't count. The exception is a pole whose
    experiments are all set aside, as in the standing rule. The count, reviewers and recency
    cover every experiment on the neuron.
    """
    from openinterpretability.builder.wiki import (
        load_archive_paths,  # Jinja: keep the CLI start light
    )
    from openinterpretability.core.errata import load_errata

    pairs = load_archive_paths(Path(wiki_dir), skip_missing=True)  # oldest first
    return states_of(pairs, load_errata(wiki_dir))


def states_of(pairs, errata) -> dict[str, NeuronState]:
    """The states of :func:`load_states`, from records that are already loaded.

    ``pairs`` are ``(path, record)`` tuples, oldest first, as ``load_archive_paths`` returns
    them. The site build uses this function, so that it does not load the archive again. A
    second copy of every record made the deploy runner run out of memory.
    """
    from openinterpretability.core.errata import corrected

    states: dict[str, NeuronState] = {}
    records = []
    counted: set[str] = set()  # poles with an experiment that isn't set aside
    for path, raw in pairs:
        record, aside = corrected(raw, errata)
        records.append(raw)
        slug = record.submission.identifier.slug
        state = states.setdefault(slug, NeuronState(slug, record, path))
        if not aside or slug not in counted:
            state.record, state.record_path = record, path
        if not aside:
            counted.add(slug)
        state.experiments += 1
        state.reviewers.add(canonical_agent(record.verdict.reviewer_agent_model))
        state.independent_reviews += not record.self_reviewed
        state.last = record.verdict.timestamp
    for slug, standing in standings(records, errata).items():
        # Every standing's lead is a record that the loop above already added, so this always
        # holds. It guards against a change to how standings are built.
        if slug in states:  # pragma: no branch
            state = states[slug]
            state.status, state.standing_record = standing.status, standing.lead.record
            state.contested = standing.contested
            state.contested_hypothesis = standing.contested_by_hypothesis
    return states


def standings(records, errata) -> dict:
    """Map the slug of each native pole and head to its standing (``builder.standing``)."""
    from openinterpretability.builder.standing import native_standings
    from openinterpretability.builder.tree import build_tree

    return {
        standing.lead.record.submission.identifier.slug: standing
        for _, _, standing in native_standings(build_tree(records, errata))
    }
