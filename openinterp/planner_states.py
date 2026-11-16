"""What the planner knows of each studied unit: its latest counted record and its history.

:func:`load_states` reads the archive with the errata of each record applied
(:mod:`openinterp.core.errata`, issue #3271). So the planner sees the standings of the site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openinterp.core.agent_names import canonical_agent
from openinterp.core.schemas import AuditRecord


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
    contested: bool = False  # its readings split evenly between VERIFIED and not (#3212)


def load_states(wiki_dir: Path) -> dict[str, NeuronState]:
    """One state per neuron pole, derived from all of its archived experiments.

    Each record has the edits of its errata applied (:mod:`openinterp.core.errata`).
    Follow-ups are planned from the most recent experiment that counts toward the status.
    An experiment that an erratum sets aside doesn't count. The exception is a pole whose
    experiments are all set aside, as in the standing rule. The count, reviewers and recency
    cover every experiment on the neuron.
    """
    from openinterp.builder.wiki import load_archive_paths  # Jinja: keep the CLI start light
    from openinterp.core.errata import corrected, load_errata

    errata = load_errata(wiki_dir)
    states: dict[str, NeuronState] = {}
    records = []
    counted: set[str] = set()  # poles with an experiment that isn't set aside
    for path, raw in load_archive_paths(Path(wiki_dir)):  # oldest first
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
    for slug in contested_slugs(records, errata):
        if slug in states:
            states[slug].contested = True
    return states


def contested_slugs(records, errata) -> set[str]:
    """The slugs of the poles whose standing is CONTESTED (``builder.standing``, #3212)."""
    from openinterp.builder.standing import contested_units
    from openinterp.builder.tree import build_tree

    found = set()
    for row in contested_units(build_tree(records, errata)):
        found.add(row["standing"].lead.record.submission.identifier.slug)
    return found
