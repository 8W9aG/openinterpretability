"""Group every archived experiment into Model → Layer → Neuron → Experiments.

All experiments are published, whatever their verdict. Within a neuron they are
ordered by how conclusive the result is (see :mod:`significance`), so a strong null
result sits next to a strong positive one rather than being hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from openinterp.builder.significance import Significance, significance
from openinterp.builder.wiki import experiment_id
from openinterp.core.schemas import AuditRecord


@dataclass
class Experiment:
    record: AuditRecord
    sig: Significance
    id: str
    is_latest: bool  # newest experiment for this neuron pole
    reuses: str | None = None  # id of an earlier experiment with the same probe set

    @property
    def independent(self) -> bool:
        """False when this re-measures an earlier experiment's probes (e.g. a re-evaluation):
        the two are not independent evidence and must not be counted twice."""
        return self.reuses is None

    @property
    def anchor(self) -> str:
        return f"{self.record.submission.identifier.pole}-{self.id}"


@dataclass
class NeuronNode:
    model: str
    layer: int
    idx: int
    href: str
    experiments: list[Experiment] = field(default_factory=list)

    @property
    def latest_by_pole(self) -> dict[str, Experiment]:
        return {e.record.submission.identifier.pole: e for e in self.experiments if e.is_latest}

    @property
    def verified(self) -> bool:
        return any(e.record.is_verified for e in self.latest_by_pole.values())

    @property
    def top(self) -> Experiment:
        return self.experiments[0]


@dataclass
class LayerNode:
    model: str
    layer: int
    href: str
    neurons: list[NeuronNode] = field(default_factory=list)

    @property
    def experiments(self) -> int:
        return sum(len(n.experiments) for n in self.neurons)

    @property
    def verified(self) -> int:
        return sum(n.verified for n in self.neurons)


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


def layer_href(slug: str, layer: int) -> str:
    return f"models/{slug}/L{layer}/index.html"


def neuron_href(slug: str, layer: int, idx: int) -> str:
    return f"models/{slug}/L{layer}/N{idx}.html"


def experiment_order(e: Experiment) -> tuple:
    """Most conclusive first; on ties independent evidence before re-measurements, then
    newest first, then deterministically."""
    ident = e.record.submission.identifier
    return (
        -e.sig.conclusiveness,
        not e.independent,
        -e.record.verdict.timestamp.timestamp(),
        ident.pole,
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
    neurons: dict[tuple[str, int, int], NeuronNode] = {}
    for r in reviewed:
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
        key = (ident.model_name, ident.layer, ident.neuron_idx)
        if key not in neurons:
            neurons[key] = NeuronNode(
                ident.model_name,
                ident.layer,
                ident.neuron_idx,
                neuron_href(mslug, ident.layer, ident.neuron_idx),
            )
        neurons[key].experiments.append(
            Experiment(r, significance(r), experiment_id(r), latest[ident.slug] is r)
        )

    for (name, layer_idx, _), node in sorted(neurons.items()):
        _mark_reuse(node.experiments)
        node.experiments.sort(key=experiment_order)
        model = models[name]
        if not model.layers or model.layers[-1].layer != layer_idx:
            model.layers.append(LayerNode(name, layer_idx, layer_href(model.slug, layer_idx)))
        model.layers[-1].neurons.append(node)
    return [models[name] for name in sorted(models)]


def _mark_reuse(experiments: list[Experiment]) -> None:
    first: dict[tuple, str] = {}
    for e in sorted(experiments, key=lambda e: e.record.verdict.timestamp):
        probes = frozenset(
            (r.prompt, r.expected_active)
            for r in e.record.verdict.adversarial_test_results
            if r.kind in ("hard_negative", "positive_holdout")
        )
        key = (e.record.submission.identifier.pole, probes)
        if probes and key in first:
            e.reuses = first[key]
        else:
            first.setdefault(key, e.id)


def false_discovery(models: Sequence[ModelNode]) -> dict:
    """Expected false discoveries among the current VERIFIED claims (one per neuron pole).

    Each claim is false with probability 1 − P(useful), so the expected number of
    false claims is their sum and the estimated false discovery rate is that over the
    number of claims.
    """
    from openinterp.core.stats import expected_false_discoveries

    claims = [
        e.sig.p_useful
        for m in models
        for layer in m.layers
        for n in layer.neurons
        for e in n.latest_by_pole.values()
        if e.record.is_verified
    ]
    expected = expected_false_discoveries(claims)
    return {
        "claims": len(claims),
        "expected_false": round(expected, 2),
        "rate": round(expected / len(claims), 4) if claims else None,
    }
