"""Group every archived experiment into Model → Layer → Unit → Experiments.

A unit is an MLP neuron (both poles on one page), an attention head, or a latent of a
sparse dictionary (the learned-basis baseline, kept apart from native-unit counts).

All experiments are published, whatever their verdict. Within a neuron they are
ordered by how conclusive the result is (see :mod:`significance`), so a strong null
result sits next to a strong positive one rather than being hidden.

A pole's status is derived from its experiments by :mod:`standing`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openinterp.builder.significance import Significance, significance
from openinterp.builder.standing import native_standings, pole_standing
from openinterp.builder.wiki import experiment_id
from openinterp.core.schemas import AuditRecord

if TYPE_CHECKING:
    from openinterp.builder.standing import Standing


@dataclass
class Experiment:
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

    @property
    def independent(self) -> bool:
        """False when this re-measures an earlier experiment's probes (e.g. a re-evaluation):
        the two are not independent evidence and must not be counted twice."""
        return self.reuses is None

    @property
    def side(self) -> str:
        """A neuron experiment's pole, else ``head`` or ``feature``: what the unit page
        groups it under."""
        ident = self.record.submission.identifier
        return ident.pole or SIDES[ident.kind][0]

    @property
    def anchor(self) -> str:
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
    """One neuron (both poles), attention head or dictionary feature, with all of its
    experiments."""

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
        return self.kind == "attn_head"

    @property
    def is_neuron(self) -> bool:
        return self.kind == "mlp_neuron_pole"

    @property
    def is_feature(self) -> bool:
        return self.kind == "dictionary_feature"

    @property
    def sides(self) -> tuple[str, ...]:
        return SIDES[self.kind]

    @property
    def noun(self) -> str:
        return NOUNS[self.kind]

    @property
    def label(self) -> str:
        return f"{LABEL_PREFIX[self.kind]}{self.idx}"

    @property
    def verified(self) -> bool:
        return any(s.verified for s in self.standings.values())

    @property
    def top(self) -> Experiment:
        return self.experiments[0]


@dataclass
class LayerNode:
    model: str
    layer: int
    href: str
    units: list[UnitNode] = field(default_factory=list)  # neurons, then heads, then features

    @property
    def neurons(self) -> list[UnitNode]:
        return [u for u in self.units if u.is_neuron]

    @property
    def heads(self) -> list[UnitNode]:
        return [u for u in self.units if u.is_head]

    @property
    def features(self) -> list[UnitNode]:
        return [u for u in self.units if u.is_feature]

    @property
    def experiments(self) -> int:
        return sum(len(n.experiments) for n in self.units)

    @property
    def verified(self) -> int:
        """Neurons with a verified pole."""
        return sum(n.verified for n in self.neurons)

    @property
    def verified_heads(self) -> int:
        return sum(n.verified for n in self.heads)

    @property
    def verified_features(self) -> int:
        return sum(n.verified for n in self.features)


@dataclass
class ModelNode:
    name: str
    slug: str
    n_layers: int
    href: str
    mlp_width: int | None = None  # neurons per layer; None when no record says
    layers: list[LayerNode] = field(default_factory=list)

    @property
    def neurons(self) -> int:
        return sum(len(layer.neurons) for layer in self.layers)

    @property
    def heads(self) -> int:
        return sum(len(layer.heads) for layer in self.layers)

    @property
    def verified_heads(self) -> int:
        return sum(layer.verified_heads for layer in self.layers)

    @property
    def features(self) -> int:
        return sum(len(layer.features) for layer in self.layers)

    @property
    def verified_features(self) -> int:
        return sum(layer.verified_features for layer in self.layers)

    @property
    def experiments(self) -> int:
        return sum(layer.experiments for layer in self.layers)

    @property
    def verified(self) -> int:
        return sum(layer.verified for layer in self.layers)

    @property
    def total_neurons(self) -> int | None:
        return self.n_layers * self.mlp_width if self.mlp_width else None

    @property
    def coverage(self) -> float | None:
        """Share of the model's neurons with at least one experiment."""
        return self.neurons / self.total_neurons if self.total_neurons else None

    @property
    def verified_share(self) -> float | None:
        """Share of the model's neurons with a verified pole."""
        return self.verified / self.total_neurons if self.total_neurons else None

    def layer_counts(self) -> list[int]:
        """Studied neurons per layer index, including unstudied layers (0)."""
        counts = {layer.layer: len(layer.neurons) for layer in self.layers}
        return [counts.get(i, 0) for i in range(self.n_layers)]


# MLP width of models whose early records predate ``model_mlp_width``. Records are
# write-once, so the width can't be backfilled into them; a width on any record wins.
KNOWN_MLP_WIDTHS = {
    "Qwen/Qwen2.5-0.5B": 4864,
}


def model_href(slug: str) -> str:
    return f"models/{slug}/index.html"


def circuits_href(slug: str) -> str:
    return f"models/{slug}/circuits.html"


def layer_href(slug: str, layer: int) -> str:
    return f"models/{slug}/L{layer}/index.html"


def neuron_href(slug: str, layer: int, idx: int) -> str:
    return f"models/{slug}/L{layer}/N{idx}.html"


def head_href(slug: str, layer: int, idx: int) -> str:
    return f"models/{slug}/L{layer}/H{idx}.html"


def feature_href(slug: str, layer: int, dictionary: str, idx: int) -> str:
    return f"models/{slug}/L{layer}/{dictionary}/F{idx}.html"


def unit_href(ident) -> str:
    """The page of the neuron, head or feature ``ident`` names."""
    if ident.kind == "dictionary_feature":
        return feature_href(ident.model_slug, ident.layer, ident.dictionary, ident.index)
    href = head_href if ident.kind == "attn_head" else neuron_href
    return href(ident.model_slug, ident.layer, ident.index)


def experiment_order(e: Experiment) -> tuple:
    """Most conclusive first; on ties independent evidence before re-measurements, then
    newest first, then deterministically."""
    return (
        -e.sig.conclusiveness,
        not e.independent,
        -e.record.verdict.timestamp.timestamp(),
        e.side,
        e.id,
    )


def build_tree(records: Sequence[AuditRecord]) -> list[ModelNode]:
    reviewed = [r for r in records if r.verdict is not None]
    latest: dict[str, AuditRecord] = {}
    for r in reviewed:
        slug = r.submission.identifier.slug
        if slug not in latest or r.verdict.timestamp > latest[slug].verdict.timestamp:
            latest[slug] = r

    models: dict[str, ModelNode] = {}
    units: dict[tuple[str, int, int, str, int], UnitNode] = {}  # neurons, heads, features
    for r in reviewed:
        _add_experiment(models, units, r, latest[r.submission.identifier.slug] is r)

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


def _add_experiment(models: dict, units: dict, r: AuditRecord, is_latest: bool) -> None:
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
            KNOWN_MLP_WIDTHS.get(ident.model_name),
        )
    model.n_layers = max(model.n_layers, r.submission.model_num_layers or 0, ident.layer + 1)
    if r.submission.model_mlp_width:
        model.mlp_width = r.submission.model_mlp_width
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
    units[key].experiments.append(
        Experiment(
            r,
            significance(r),
            experiment_id(r),
            is_latest,
            claim=r.submission.model_dump_json(exclude={"provenance"}),
        )
    )


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
    """Expected false discoveries among the current VERIFIED claims about native units (one
    per neuron pole or head; dictionary features are a baseline, not claims about the model's
    own units).

    Each claim is false with probability 1 − P(useful), so the expected number of
    false claims is their sum and the estimated false discovery rate is that over the
    number of claims.
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
