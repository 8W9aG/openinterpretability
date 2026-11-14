"""Errata: harness bugs that affected signed records (``docs/PROTOCOL.md`` §5).

Records cannot be rewritten. A maintainer who finds a bug that affected some of them
appends an erratum to ``wiki/errata.jsonl``. Then they sign the file with
``open-interpretability sign wiki/errata.jsonl``. Each line is one :class:`Erratum`. It names the
records it covers (a list of record paths, a :class:`ErratumRule`, or both), the bug, and
the commit that fixed it. It either lists :class:`Edit` objects for the records, or sets
the records aside.

The record files never change. The site build applies the edits to a copy of each covered
record (:func:`apply_errata`). So pages, standings and ``registry.parquet`` read the
corrected values. ``open-interpretability reproduce`` checks the corrected record, and skips a
record that is set aside.

``open-interpretability check`` verifies the file (:mod:`openinterp.errata_checks`). Errata have
no page of their own. The site shows each erratum on the experiments it covers, with the old
and new value of each edit. The standing rule (:mod:`openinterp.builder.standing`) leaves
out the records that are set aside.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from openinterp.core.paths import DEFAULT_WIKI_DIR
from openinterp.core.strict import StrictModel

if TYPE_CHECKING:
    from openinterp.core.schemas import AuditRecord

ERRATA_FILE = "errata.jsonl"
"""The errata file's name in the wiki directory."""

FIXED_PATHS = ("submission.identifier", "verdict.timestamp")
"""Fields an edit can't change: they give the record's unit and id, so its place."""


class Edit(StrictModel):
    """One corrected field of a record: its dotted path and its new value.

    The path names keys and list indices of the record's JSON, for example
    ``verdict.status`` or ``submission.top_examples.0.max_activation``. The field must exist.
    """

    path: str = Field(pattern=r"^[a-z_]+(\.([a-z_]+|[0-9]+))*$")
    value: JsonValue

    @model_validator(mode="after")
    def _not_fixed(self) -> Edit:
        if any(self.path == p or self.path.startswith(p + ".") for p in FIXED_PATHS):
            raise ValueError(f"an edit can't change {self.path}: it places the record")
        return self


@dataclass(frozen=True)
class AppliedEdit:
    """An edit as applied to one record: which erratum made it, and the value before."""

    erratum: str  # the erratum's id
    path: str
    before: Any
    after: Any


class ErratumRule(StrictModel):
    """A rule that selects records. A record matches when it matches every field given.

    A field that is ``None`` matches every record. A record without a review policy has no
    protocol version, so a rule with ``protocol_versions`` does not match it.
    """

    models: list[str] | None = Field(default=None, min_length=1, description="Model names.")
    kinds: list[Literal["mlp_neuron_pole", "attn_head", "dictionary_feature"]] | None = Field(
        default=None, min_length=1
    )
    protocol_versions: list[str] | None = Field(default=None, min_length=1)
    harness_versions: list[str] | None = Field(default=None, min_length=1)
    reviewed_from: AwareDatetime | None = Field(
        default=None, description="First verdict time that matches (inclusive)."
    )
    reviewed_before: AwareDatetime | None = Field(
        default=None, description="Verdict time from which records no longer match."
    )

    @model_validator(mode="after")
    def _selects_something(self) -> ErratumRule:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("a rule needs at least one field; list the records instead")
        return self

    def matches(self, record: AuditRecord) -> bool:
        """Return whether ``record`` matches every field of the rule."""
        ident, verdict = record.submission.identifier, record.verdict
        provenance = record.submission.provenance
        policy = verdict.policy if verdict is not None else None
        checks = (
            (self.models, ident.model_name),
            (self.kinds, ident.kind),
            (self.protocol_versions, policy.protocol_version if policy else None),
            (self.harness_versions, provenance.harness_version if provenance else None),
        )
        if any(allowed is not None and value not in allowed for allowed, value in checks):
            return False
        when = verdict.timestamp if verdict is not None else None
        if self.reviewed_from is not None and (when is None or when < self.reviewed_from):
            return False
        return self.reviewed_before is None or (when is not None and when < self.reviewed_before)


class Erratum(StrictModel):
    """One harness bug and the records it affected. One line of ``wiki/errata.jsonl``."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", description="Unique; the site's anchor.")
    date: Date
    title: str = Field(min_length=1, description="One line for the banner.")
    bug: str = Field(min_length=1, description="What was wrong, and what it did to records.")
    fixed_in: str = Field(pattern=r"^[0-9a-f]{7,40}$", description="The commit that fixed it.")
    edits: list[Edit] = Field(default_factory=list, description="Corrections to each record.")
    set_aside: bool = Field(default=False, description="The records count toward no status.")
    records: list[str] = Field(
        default_factory=list,
        description="Covered records, as paths relative to wiki/experiments/.",
    )
    rule: ErratumRule | None = None

    @model_validator(mode="after")
    def _covers_something(self) -> Erratum:
        if not self.records and self.rule is None:
            raise ValueError("an erratum needs records, a rule, or both")
        if bool(self.edits) == self.set_aside:
            raise ValueError("an erratum either lists edits or sets its records aside")
        return self

    def covers(self, record: AuditRecord) -> bool:
        """Return whether the erratum covers ``record``: it is listed or matches the rule."""
        return record_ref(record) in self.records or (
            self.rule is not None and self.rule.matches(record)
        )


def errata_path(wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> Path:
    """Where the errata file is: ``<wiki_dir>/errata.jsonl``."""
    return Path(wiki_dir) / ERRATA_FILE


def record_ref(record: AuditRecord) -> str:
    """How an erratum names ``record``: its path relative to ``wiki/experiments/``."""
    from openinterp.builder.wiki import archive_path
    from openinterp.core.paths import experiments_dir

    return archive_path(Path(), record).relative_to(experiments_dir(Path())).as_posix()


def parse_errata(text: str) -> list[Erratum]:
    """Parse the content of an errata file. Raises ``ValueError`` naming the bad line."""
    errata = []
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            errata.append(Erratum.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc
    return errata


def load_errata(wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> list[Erratum]:
    """Every erratum in ``wiki_dir``, in file order; none if there is no errata file."""
    path = errata_path(wiki_dir)
    return parse_errata(path.read_text(encoding="utf-8")) if path.exists() else []


def covering(errata: Sequence[Erratum], record: AuditRecord) -> list[Erratum]:
    """The errata in ``errata`` that cover ``record``, in file order."""
    return [erratum for erratum in errata if erratum.covers(record)]


def apply_errata(
    record: AuditRecord, errata: Sequence[Erratum]
) -> tuple[AuditRecord, list[AppliedEdit]]:
    """``record`` with the edits of ``errata`` applied in order, and what each edit changed.

    Returns ``record`` itself when no erratum edits it. Raises ``ValueError`` when an edit's
    path is not in the record, or the edited record is not valid.
    """
    from openinterp.core.schemas import AuditRecord

    edits = [(erratum.id, edit) for erratum in errata for edit in erratum.edits]
    if not edits:
        return record, []
    data = record.model_dump(mode="json")
    applied = []
    for erratum_id, edit in edits:
        before = _replace(data, edit.path, edit.value)
        applied.append(AppliedEdit(erratum_id, edit.path, before, edit.value))
    try:
        return AuditRecord.model_validate_json(json.dumps(data)), applied
    except ValueError as exc:
        raise ValueError(f"the edited record is not valid: {exc}") from exc


def _replace(data: Any, path: str, value: Any) -> Any:
    """Set the field at dotted ``path`` in ``data`` to ``value``; return its old value."""
    *parents, last = path.split(".")
    node = data
    try:
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key = int(last) if isinstance(node, list) else last
        before = node[key]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError(f"the record has no field {path}") from None
    node[key] = value
    return before
