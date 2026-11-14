"""Group every archived experiment into Model → Layer → Unit → Experiments.

A unit is an MLP neuron (both poles on one page), an attention head, a latent of a
sparse dictionary or a superposition. Latents are the learned-basis baseline. A
superposition is a sparse direction over one layer's neurons. Both are kept apart from
native-unit counts.

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

from openinterp.builder.experiment import (  # noqa: F401  (re-exported)
    LABEL_PREFIX,
    NATIVE_KINDS,
    NOUNS,
    SIDES,
    Experiment,
    unit_side,
)
from openinterp.builder.hrefs import (  # noqa: F401  (re-exported: pages import them from here)
    circuits_href,
    experiment_href,
    feature_href,
    head_href,
    layer_href,
    model_href,
    neuron_href,
    unit_href,
)
from openinterp.builder.significance import significance
from openinterp.builder.standing import native_standings, pole_standing
from openinterp.builder.wiki import experiment_id
from openinterp.core.errata import Erratum, apply_errata, covering, record_ref
from openinterp.core.schemas import AuditRecord

if TYPE_CHECKING:
    from openinterp.builder.standing import Standing
    from openinterp.builder.stopped import StoppedNeuron
    from openinterp.core.profiles import SuperpositionProfile


@dataclass
class UnitNode:
    """One unit with all of its experiments.

    The unit is a neuron (both poles), an attention head, a dictionary feature or a
    superposition. A neuron's ``part_of`` lists the superpositions that it is a member of
    (:mod:`openinterp.builder.superpositions`).
    """

    model: str
    layer: int
    idx: int
    href: str
    kind: str = "mlp_neuron_pole"
    experiments: list[Experiment] = field(default_factory=list)
    dictionary: str | None = None  # a feature's dictionary
    standings: dict[str, Standing] = field(default_factory=dict)  # per pole (or head, feature)
    superposition_id: str | None = None  # a superposition's id
    members: list = field(default_factory=list)  # a superposition's members, by |weight|
    profile: SuperpositionProfile | None = None  # a superposition's, from its newest record
    part_of: list = field(default_factory=list, repr=False, compare=False)  # a neuron's

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
    def is_superposition(self) -> bool:
        """True for a superposition: a sparse direction over one layer's neurons."""
        return self.kind == "superposition"

    @property
    def is_native(self) -> bool:
        """True for a neuron or a head: a claim about the model's own units."""
        return self.kind in NATIVE_KINDS

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
        """The unit within its layer, for example ``N7``, ``H3`` or ``S0123456789ab``."""
        return f"{LABEL_PREFIX[self.kind]}{self.superposition_id or self.idx}"

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
    units: list[UnitNode] = field(default_factory=list)  # neurons, heads, features, superpositions
    stopped: list[StoppedNeuron] = field(default_factory=list)  # by index
    members: list[UnitNode] = field(default_factory=list)  # member neurons with no experiment

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
    def pages(self) -> list[UnitNode]:
        """The layer's units with a page: its units, then its member neurons with no experiment."""
        return [*self.units, *self.members]

    @property
    def superpositions(self) -> list[UnitNode]:
        """The layer's studied superpositions."""
        return [u for u in self.units if u.is_superposition]

    @property
    def member_only(self) -> int:
        """Neurons studied only as members of a superposition: no experiment, no stopped scan."""
        stopped = {s.idx for s in self.stopped}
        return sum(n.idx not in stopped for n in self.members)

    @property
    def neurons_studied(self) -> int:
        """Studied neurons, with those studied only as members of a superposition."""
        return self.studied + self.member_only

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

    @property
    def verified_superpositions(self) -> int:
        """Superpositions with a VERIFIED standing."""
        return sum(n.verified for n in self.superpositions)


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
    def superpositions(self) -> int:
        """Studied superpositions over every layer."""
        return sum(len(layer.superpositions) for layer in self.layers)

    @property
    def verified_superpositions(self) -> int:
        """VERIFIED superpositions over every layer."""
        return sum(layer.verified_superpositions for layer in self.layers)

    @property
    def member_only(self) -> int:
        """Neurons studied only as members of a superposition, over every layer."""
        return sum(layer.member_only for layer in self.layers)

    @property
    def neurons_studied(self) -> int:
        """Studied neurons, with those studied only as members of a superposition."""
        return self.neurons + self.member_only

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
    sid = getattr(ident, "superposition_id", None)
    order = list(SIDES).index(ident.kind)
    key = (ident.model_name, ident.layer, order, dictionary or sid or "", ident.index)
    if key not in units:
        units[key] = UnitNode(
            ident.model_name,
            ident.layer,
            ident.index,
            unit_href(ident),
            ident.kind,
            dictionary=dictionary,
            superposition_id=sid,
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

    There is one claim per native neuron pole or head. Dictionary features and
    superpositions are left out. They are not claims about one of the model's own units.

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
