"""The kinds of unit the site shows, and one experiment as the site places it.

:mod:`openinterp.builder.tree` groups these experiments into models, layers and units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openinterp.builder.hrefs import experiment_href
from openinterp.builder.significance import Significance
from openinterp.core.errata import AppliedEdit, Erratum
from openinterp.core.schemas import AuditRecord

if TYPE_CHECKING:
    from openinterp.builder.standing import Standing

# What a unit page groups experiments under, per kind, and the prefix of a unit's label.
SIDES = {
    "mlp_neuron_pole": ("positive", "negative"),
    "attn_head": ("head",),
    "dictionary_feature": ("feature",),
    "superposition": ("superposition",),
}
LABEL_PREFIX = {
    "mlp_neuron_pole": "N",
    "attn_head": "H",
    "dictionary_feature": "F",
    "superposition": "S",
}
NOUNS = {
    "mlp_neuron_pole": "neuron",
    "attn_head": "head",
    "dictionary_feature": "feature",
    "superposition": "superposition",
}
NATIVE_KINDS = ("mlp_neuron_pole", "attn_head")  # the model's own units: claims, not baselines


def unit_side(ident) -> str:
    """What a unit page groups ``ident`` under: a neuron's pole, else the kind's one side."""
    return ident.pole or SIDES[ident.kind][0]


@dataclass
class Experiment:
    """One experiment, with what the site needs to place it."""

    record: AuditRecord
    sig: Significance
    id: str
    is_latest: bool  # newest experiment for this neuron pole
    reuses: str | None = None  # id of an earlier experiment with the same probe set
    claim: str = ""  # the submission without its provenance: equal for re-reviews of one hypothesis
    role: str = "current"  # what it counts for in the pole's standing (standing.ROLES)
    # Revision links (``supersedes``) point both ways, so keep them out of repr and ==.
    revises: Experiment | None = field(default=None, repr=False, compare=False)
    revised_by: list[Experiment] = field(default_factory=list, repr=False, compare=False)
    standing: Standing | None = field(default=None, repr=False, compare=False)  # its pole's
    errata: list[Erratum] = field(default_factory=list, repr=False, compare=False)  # covering it
    edits: list[AppliedEdit] = field(default_factory=list, repr=False, compare=False)  # theirs
    user: str | None = None  # who made it: its contributor, else its PR's author (users.py)

    @property
    def set_aside(self) -> bool:
        """Whether an erratum sets this experiment aside, so it counts toward no standing."""
        return any(erratum.set_aside for erratum in self.errata)

    @property
    def independent(self) -> bool:
        """False when this re-measures an earlier experiment's probes.

        An example is a re-evaluation. The two are not independent evidence, and must not
        be counted twice.
        """
        return self.reuses is None

    @property
    def side(self) -> str:
        """What the unit page groups the experiment under.

        That is a neuron experiment's pole, else ``head`` or ``feature``.
        """
        return unit_side(self.record.submission.identifier)

    @property
    def anchor(self) -> str:
        """The experiment's old anchor on its unit's page, which now redirects to its page."""
        return f"{self.side}-{self.id}"

    @property
    def href(self) -> str:
        """The experiment's own page."""
        return experiment_href(self.record.submission.identifier.slug, self.id)

    @property
    def probe_set(self) -> str:
        """The experiment whose probes this one measured: re-measurements share it."""
        return self.reuses or self.id

    @property
    def counts(self) -> bool:
        """Whether the pole's current status rests on this experiment."""
        return self.role == "current"
