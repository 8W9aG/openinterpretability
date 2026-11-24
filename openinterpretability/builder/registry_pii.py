"""Remove personal data and taken-down text from the registry's records, for the site.

Records keep their text unchanged, because experiments study this text, and records are
signed. Only the build's output changes. :mod:`labinacan.site.pii` does the redaction.
This module applies it to the registry's records, their attribution and their errata.

It also tells the redaction core this project's settings. The matches of each text are
kept on disk, in ``$XDG_CACHE_HOME/openinterpretability/pii``
(``$OPENINTERPRETABILITY_PII_CACHE`` moves it, and ``off`` turns it off).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as dc_replace
from typing import Any

from labinacan.site import pii
from labinacan.site.pii import REMOVED_TOKEN, redact_text, scrub, scrub_field
from openinterpretability.core.cache_dirs import cache_root
from openinterpretability.core.confounds import PII_DETECTOR
from openinterpretability.core.schemas import AuditRecord

CACHE_ENV = "OPENINTERPRETABILITY_PII_CACHE"
pii.configure(lambda: cache_root(CACHE_ENV, "pii"), PII_DETECTOR)  # #6628


def _scrub_attribution(submission):
    """Remove attribution tokens that a match touched in the top example they come from.

    Attribution stores loose tokens, so the search cannot read them as one text.
    """
    attribution = submission.input_attribution
    if attribution is None:
        return submission

    def at(context, offset: int, token: str) -> str:
        if context.example >= len(submission.top_examples):
            return token
        example = submission.top_examples[context.example]
        position = example.max_token_index + offset
        removed = 0 <= position < len(example.tokens) and example.tokens[position] == REMOVED_TOKEN
        return REMOVED_TOKEN if removed else token

    examples = [
        context.model_copy(
            update={
                "peak_token": at(context, 0, context.peak_token),
                "top": [
                    t.model_copy(update={"token": at(context, t.offset, t.token)})
                    for t in context.top
                ],
            }
        )
        for context in attribution.examples
    ]
    return submission.model_copy(
        update={"input_attribution": attribution.model_copy(update={"examples": examples})}
    )


def scrub_errata(errata: Sequence[Any], edits: Sequence[Any]) -> tuple[list, list]:
    """Return what a page shows of its errata, with personal data and taken-down text removed.

    ``errata`` are :class:`openinterpretability.core.errata.Erratum`: their title and bug text are
    redacted. ``edits`` are :class:`openinterpretability.core.errata.AppliedEdit`. An edit can
    change a text field of a record, so its values before and after are scrubbed. An edit to a
    token list (a path that ends in ``.tokens``) gets the joined search of :func:`redact_tokens`.
    """
    notes = [
        e.model_copy(update={"title": redact_text(e.title), "bug": redact_text(e.bug)})
        for e in errata
    ]
    changes = [
        dc_replace(
            d,
            before=scrub_field(d.path.rpartition(".")[2], d.before),
            after=scrub_field(d.path.rpartition(".")[2], d.after),
        )
        for d in edits
    ]
    return notes, changes


def scrub_experiments(experiments: Sequence[Any]) -> None:
    """Scrub, in place, what each experiment's page shows: its record and its errata notes.

    ``experiments`` are :class:`openinterpretability.builder.tree.Experiment`.
    """
    for e in experiments:
        e.record = scrub_record(e.record)
        e.scrubbed = True  # a full record loaded later is scrubbed too (builder/slim.py)
        e.errata, e.edits = scrub_errata(e.errata, e.edits)


def scrub_record(record: AuditRecord) -> AuditRecord:
    """Return a copy of ``record`` to publish, with personal data removed from all its text.

    The copy reaches every page and the Parquet export, including fields added later. Call
    this after computing anything from probe text. The record on disk does not change.
    """
    scrubbed = scrub(record)
    return scrubbed.model_copy(update={"submission": _scrub_attribution(scrubbed.submission)})
