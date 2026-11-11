"""Neurons that only stopped scans cover: studied, but with no experiment.

A scan stops on a neuron pole that is dead or default-on (:mod:`openinterp.core.draws`).
The neuron still counts as studied, in the site's counts and maps. When the weights prove
the pole dead, it also counts as verified (:func:`openinterp.builder.tree.explore_focus`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from openinterp.builder.tree import LayerNode, ModelNode, layer_href


@dataclass
class StoppedNeuron:
    """A neuron whose scans all stopped, so that neither pole has an experiment.

    ``outcomes`` gives each stopped pole and why, for example ``negative default-on``.
    """

    layer: int
    idx: int
    outcomes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """The neuron within its layer, for example ``N7``."""
        return f"N{self.idx}"


STOP_OUTCOMES = {"dead": "dead", "default_on": "default-on"}


def add_stopped(models: Sequence[ModelNode], stops: Sequence) -> None:
    """File each neuron that only stopped scans cover under its model and layer.

    ``stops`` are the random draws and the stops of chosen units (:mod:`openinterp.core.draws`).
    A neuron with an experiment on either pole is already a unit, so it is left out. So is
    a model with no experiment: its width is unknown.
    """
    by_name = {m.name: m for m in models}
    found: dict[tuple[str, int, int], set[str]] = {}
    for stop in stops:
        ident = stop.identifier
        if ident.kind != "mlp_neuron_pole" or ident.model_name not in by_name:
            continue
        why = "proven dead" if stop.proven_dead else STOP_OUTCOMES[stop.outcome]
        key = (ident.model_name, ident.layer, ident.neuron_idx)
        found.setdefault(key, set()).add(f"{ident.pole} {why}")
    for (name, layer_idx, idx), outcomes in sorted(found.items()):
        model = by_name[name]
        layer = next((lay for lay in model.layers if lay.layer == layer_idx), None)
        if layer is not None and any(n.idx == idx for n in layer.neurons):
            continue
        if layer is None:
            layer = LayerNode(name, layer_idx, layer_href(model.slug, layer_idx))
            model.layers.append(layer)
            model.layers.sort(key=lambda lay: lay.layer)
            model.n_layers = max(model.n_layers, layer_idx + 1)
        layer.stopped.append(StoppedNeuron(layer_idx, idx, sorted(outcomes)))
