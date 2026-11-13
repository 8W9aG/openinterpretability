"""Group every archived experiment into Model → Layer → Unit → Experiments.

A unit is an MLP neuron (both poles on one page), an attention head, or a latent of a
sparse dictionary. Latents are the learned-basis baseline, kept apart from native-unit
counts.

All experiments are published, whatever their verdict. Within a neuron, they are
ordered by how conclusive the result is (see :mod:`significance`). So a strong null
result sits next to a strong positive one, and is not hidden.

A pole's status is derived from its experiments by :mod:`standing`.

A neuron whose scans all stopped (dead or default-on) has no experiment. It still counts
as studied, as a stopped neuron of its layer (:mod:`openinterp.builder.stopped`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openinterp.builder.hrefs import (  # noqa: F401  (re-exported: pages import them from here)
    circuits_href,
    feature_href,
    head_href,
    layer_href,
    model_href,
    neuron_href,
)
from openinterp.builder.significance import Significance, significance
from openinterp.builder.standing import native_standings, pole_standing
from openinterp.builder.wiki import experiment_id
from openinterp.core.errata import AppliedEdit, Erratum, apply_errata, covering, record_ref
from openinterp.core.schemas import AuditRecord

if TYPE_CHECKING:
    from openinterp.builder.standing import Standing
    from openinterp.builder.stopped import StoppedNeuron


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
        """The experiment's anchor on its unit's page."""
        return f"{self.side}-{self.id}"

    @property
    def probe_set(self) -> str:
        """The experiment whose probes this one measured: re-measurements share it."""
        return self.reuses or self.id

    @property
    def counts(self) -> bool:
        """Whether the pole's current status rests on this experiment."""
        return self.role == "current"


# What a unit page groups experiments under, per kind, and the prefix of a unit's label.
SIDES = {
    "mlp_neuron_pole": ("positive", "negative"),
    "attn_head": ("head",),
    "dictionary_feature": ("feature",),
}
LABEL_PREFIX = {"mlp_neuron_pole": "N", "attn_head": "H", "dictionary_feature": "F"}
NOUNS = {"mlp_neuron_pole": "neuron", "attn_head": "head", "dictionary_feature": "feature"}


@dataclass
class UnitNode:
    """One unit with all of its experiments.

    The unit is a neuron (both poles), an attention head or a dictionary feature.
    """

    model: str
    layer: int
    idx: int
    href: str
    kind: str = "mlp_neuron_pole"
    experiments: list[Experiment] = field(default_factory=list)
    dictionary: str | None = None  # a feature's dictionary
    standings: dict[str, Standing] = field(default_factory=dict)  # per pole (or head, feature)

    @property
    def is_head(self) -> bool:
        """True for an attention head."""
        return self.kind == "attn_head"

    @property
    def is_neuron(self) -> bool:
        """True for an MLP neuron."""
        return self.kind == "mlp_neuron_pole"

    @property
    def is_feature(self) -> bool:
        """True for a dictionary feature."""
        return self.kind == "dictionary_feature"

    @property
    def neuronpedia(self) -> str | None:
        """The feature's Neuronpedia page, where Neuronpedia hosts it; ``None`` otherwise."""
        from openinterp.crossref import latent_url

        if not self.is_feature or self.dictionary is None:
            return None
        return latent_url(self.dictionary, self.layer, self.idx)

    @property
    def sides(self) -> tuple[str, ...]:
        """The sides that the unit page groups experiments under."""
        return SIDES[self.kind]

    @property
    def noun(self) -> str:
        """The unit's kind as a noun, for page text."""
        return NOUNS[self.kind]

    @property
    def label(self) -> str:
        """The unit within its layer, for example ``N7`` or ``H3``."""
        return f"{LABEL_PREFIX[self.kind]}{self.idx}"

    @property
    def verified(self) -> bool:
        """True when any side of the unit stands VERIFIED."""
        return any(s.verified for s in self.standings.values())

    @property
    def top(self) -> Experiment:
        """The most conclusive experiment."""
        return self.experiments[0]


@dataclass
class LayerNode:
    """One layer of a model, with its studied units."""

    model: str
    layer: int
    href: str
    units: list[UnitNode] = field(default_factory=list)  # neurons, then heads, then features
    stopped: list[StoppedNeuron] = field(default_factory=list)  # by index

    @property
    def studied(self) -> int:
        """Studied neurons: those with an experiment, and those whose scans all stopped."""
        return len(self.neurons) + len(self.stopped)

    @property
    def neurons(self) -> list[UnitNode]:
        """The layer's studied neurons."""
        return [u for u in self.units if u.is_neuron]

    @property
    def heads(self) -> list[UnitNode]:
        """The layer's studied heads."""
        return [u for u in self.units if u.is_head]

    @property
    def features(self) -> list[UnitNode]:
        """The layer's studied dictionary features."""
        return [u for u in self.units if u.is_feature]

    @property
    def experiments(self) -> int:
        """Experiments on the layer's units."""
        return sum(len(n.experiments) for n in self.units)

    @property
    def verified(self) -> int:
        """Neurons with a verified pole."""
        return sum(n.verified for n in self.neurons)

    @property
    def verified_heads(self) -> int:
        """Heads with a VERIFIED standing."""
        return sum(n.verified for n in self.heads)

    @property
    def verified_features(self) -> int:
        """Dictionary features with a VERIFIED standing."""
        return sum(n.verified for n in self.features)


@dataclass
class ModelNode:
    """One model, with its studied layers."""

    name: str
    slug: str
    n_layers: int
    href: str
    mlp_width: int | None = None  # neurons per layer; None when no record says
    layers: list[LayerNode] = field(default_factory=list)

    @property
    def neurons(self) -> int:
        """Studied neurons over every layer, including those whose scans all stopped."""
        return sum(layer.studied for layer in self.layers)

    @property
    def heads(self) -> int:
        """Studied heads over every layer."""
        return sum(len(layer.heads) for layer in self.layers)

    @property
    def verified_heads(self) -> int:
        """VERIFIED heads over every layer."""
        return sum(layer.verified_heads for layer in self.layers)

    @property
    def features(self) -> int:
        """Studied dictionary features over every layer."""
        return sum(len(layer.features) for layer in self.layers)

    @property
    def verified_features(self) -> int:
        """VERIFIED dictionary features over every layer."""
        return sum(layer.verified_features for layer in self.layers)

    @property
    def experiments(self) -> int:
        """Experiments over every layer."""
        return sum(layer.experiments for layer in self.layers)

    @property
    def verified(self) -> int:
        """Neurons with a VERIFIED pole, over every layer."""
        return sum(layer.verified for layer in self.layers)

    @property
    def total_neurons(self) -> int | None:
        """Every MLP neuron in the model, or ``None`` when no record gives the width."""
        return self.n_layers * self.mlp_width if self.mlp_width else None

    @property
    def coverage(self) -> float | None:
        """Share of the model's neurons that are studied (see :attr:`neurons`)."""
        return self.neurons / self.total_neurons if self.total_neurons else None

    @property
    def verified_share(self) -> float | None:
        """Share of the model's neurons with a verified pole."""
        return self.verified / self.total_neurons if self.total_neurons else None

    def layer_counts(self) -> list[int]:
        """Studied neurons per layer index, including unstudied layers (0)."""
        counts = {layer.layer: layer.studied for layer in self.layers}
        return [counts.get(i, 0) for i in range(self.n_layers)]


def explore_focus(models: Sequence[ModelNode], draws: Sequence = ()) -> dict:
    """The coverage target, the explore focus, and each model's verified neurons.

    A neuron counts as verified when a pole stands VERIFIED or is proven dead in
    ``draws``. Give it the random draws and the stops of chosen units together, as the
    planner reads both (:func:`openinterp.planner.dead`). The site reads a standing and
    the planner reads the latest record. So the two can name a different focus for a time.
    """
    from openinterp.planner import coverage_target  # it reads pyproject.toml
    from openinterp.planner_models import ModelCoverage, focus_model

    dead = {
        (d.identifier.model_name, d.identifier.layer, d.identifier.neuron_idx)
        for d in draws
        if d.proven_dead
    }
    counts = {}
    for m in models:
        verified = {(lay.layer, n.idx) for lay in m.layers for n in lay.neurons if n.verified}
        proven = {(layer, idx) for name, layer, idx in dead if name == m.name}
        counts[m.name] = {"verified": len(verified | proven), "dead": len(proven)}
    target = coverage_target()
    covered = (
        ModelCoverage(m.name, m.neurons, m.total_neurons, counts[m.name]["verified"])
        for m in models
    )
    return {"target": target, "model": focus_model(covered, target), "models": counts}


# MLP width of models whose early records predate ``model_mlp_width``. Records are
# write-once, so the width can't be backfilled into them; a width on any record wins.
KNOWN_MLP_WIDTHS = {
    "Qwen/Qwen2.5-0.5B": 4864,
    "Qwen/Qwen2.5-1.5B": 8960,
}


def mlp_widths(records: Iterable[AuditRecord]) -> dict[str, int]:
    """Return the neurons per MLP layer of each model, for records that do not give a width.

    A width on any record of the model wins over ``KNOWN_MLP_WIDTHS``.
    """
    widths = dict(KNOWN_MLP_WIDTHS)
    for r in records:
        if r.submission.model_mlp_width:
            widths[r.submission.identifier.model_name] = r.submission.model_mlp_width
    return widths


def unit_side(ident) -> str:
    """What a unit page groups ``ident`` under: a neuron's pole, else ``head`` or ``feature``."""
    return ident.pole or SIDES[ident.kind][0]


def unit_href(ident) -> str:
    """The page of the neuron, head or feature ``ident`` names."""
    if ident.kind == "dictionary_feature":
        return feature_href(ident.model_slug, ident.layer, ident.dictionary, ident.index)
    href = head_href if ident.kind == "attn_head" else neuron_href
    return href(ident.model_slug, ident.layer, ident.index)


def experiment_order(e: Experiment) -> tuple:
    """The sort key of a unit's experiments: most conclusive first.

    On ties, independent evidence comes before re-measurements, then newest first, then
    a fixed order.
    """
    return (
        -e.sig.conclusiveness,
        not e.independent,
        -e.record.verdict.timestamp.timestamp(),
        e.side,
        e.id,
    )


def build_tree(
    records: Sequence[AuditRecord],
    errata: Sequence[Erratum] = (),
    users: Mapping[str, str] | None = None,
) -> list[ModelNode]:
    """Group the reviewed records into models, layers and units.

    Each experiment lists the ``errata`` that cover its record (:mod:`openinterp.core.errata`),
    and holds the record with their edits applied. Its ``user`` is the proposer's recorded
    contributor, else the record's entry in ``users`` (:func:`.users.load_users`).
    """
    reviewed = [r for r in records if r.verdict is not None]
    latest: dict[str, AuditRecord] = {}
    for r in reviewed:
        slug = r.submission.identifier.slug
        if slug not in latest or r.verdict.timestamp > latest[slug].verdict.timestamp:
            latest[slug] = r

    models: dict[str, ModelNode] = {}
    units: dict[tuple[str, int, int, str, int], UnitNode] = {}  # neurons, heads, features
    widths = mlp_widths(reviewed)
    for r in reviewed:
        found = covering(errata, r)
        edited, edits = apply_errata(r, found)
        is_latest = latest[r.submission.identifier.slug] is r
        experiment = _add_experiment(models, units, edited, is_latest, widths)
        experiment.errata, experiment.edits = found, edits
        experiment.user = r.submission.contributor or (users or {}).get(record_ref(r))

    for (name, layer_idx, *_), node in sorted(units.items()):
        _mark_reuse(node.experiments)
        _link_revisions(node.experiments)
        for side in node.sides:
            pole = [e for e in node.experiments if e.side == side]
            if pole:
                node.standings[side] = pole_standing(pole)
        node.experiments.sort(key=experiment_order)
        model = models[name]
        if not model.layers or model.layers[-1].layer != layer_idx:
            model.layers.append(LayerNode(name, layer_idx, layer_href(model.slug, layer_idx)))
        model.layers[-1].units.append(node)
    return [models[name] for name in sorted(models)]


def _add_experiment(
    models: dict, units: dict, r: AuditRecord, is_latest: bool, widths: dict[str, int]
) -> Experiment:
    """File ``r`` under its unit, creating the model and unit nodes on first sight."""
    ident = r.submission.identifier
    mslug = ident.model_slug
    model = models.get(ident.model_name)
    if model is None:
        model = models[ident.model_name] = ModelNode(
            ident.model_name,
            mslug,
            0,
            model_href(mslug),
            widths.get(ident.model_name),
        )
    model.n_layers = max(model.n_layers, r.submission.model_num_layers or 0, ident.layer + 1)
    dictionary = getattr(ident, "dictionary", None)
    order = list(SIDES).index(ident.kind)
    key = (ident.model_name, ident.layer, order, dictionary or "", ident.index)
    if key not in units:
        units[key] = UnitNode(
            ident.model_name,
            ident.layer,
            ident.index,
            unit_href(ident),
            ident.kind,
            dictionary=dictionary,
        )
    experiment = Experiment(
        r,
        significance(r),
        experiment_id(r),
        is_latest,
        claim=r.submission.model_dump_json(exclude={"provenance"}),
    )
    units[key].experiments.append(experiment)
    return experiment


def _mark_reuse(experiments: list[Experiment]) -> None:
    first: dict[tuple, str] = {}
    for e in sorted(experiments, key=lambda e: e.record.verdict.timestamp):
        probes = frozenset(
            (r.prompt, r.expected_active)
            for r in e.record.verdict.adversarial_test_results
            if r.kind in ("hard_negative", "positive_holdout")
        )
        key = (e.side, probes)
        if probes and key in first:
            e.reuses = first[key]
        else:
            first.setdefault(key, e.id)


def _link_revisions(experiments: list[Experiment]) -> None:
    by_key = {(e.side, e.id): e for e in experiments}
    for e in sorted(experiments, key=lambda e: e.record.verdict.timestamp):
        target = by_key.get((e.side, e.record.submission.supersedes))
        if target is not None:
            e.revises = target
            target.revised_by.append(e)


def false_discovery(models: Sequence[ModelNode]) -> dict:
    """Return the expected false discoveries among the current VERIFIED claims.

    There is one claim per native neuron pole or head. Dictionary features are left out:
    they are a baseline, not claims about the model's own units.

    Each claim is false with probability 1 − P(useful). So the expected number of false
    claims is their sum. The estimated false discovery rate is that sum over the number
    of claims.
    """
    from openinterp.core.stats import expected_false_discoveries

    # A claim backed by several agreeing reviews takes its weakest one's P(useful).
    claims = [
        min(e.sig.p_useful for e in s.readings)
        for _, _, s in native_standings(models)
        if s.verified
    ]
    expected = expected_false_discoveries(claims)
    return {
        "claims": len(claims),
        "expected_false": round(expected, 2),
        "rate": round(expected / len(claims), 4) if claims else None,
    }
