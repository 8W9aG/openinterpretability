"""Superpositions and circuits in ``open-interpretability explain`` (issue #3617).

A superposition is a sparse direction over one layer's MLP neurons (``data/superpositions/``).
A circuit is a set of neurons and heads with an id that survives as the registry grows
(``data/circuits/``, :mod:`openinterpretability.circuit_ids`). Both overlap with the units that make
them up, so ``explain`` lists them in their own block and doesn't add them to the total.

- *Superposition.* Let w be the direction's weights and u = w / |w|. Its attribution is
  Σ_t ((a_t − ā) · u)(∂logit/∂a_t · u), at the positions where a_t · w > 0. The check
  removes that component of the neuron offset there.
- *Circuit.* Its attribution is the sum of its members' attributions, with both poles of
  each neuron member. The check mean-ablates every member together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

HEAD = "head"  # the same marker as openinterpretability.explain.HEAD, which imports this module


@dataclass(frozen=True)
class GroupDef:
    """One superposition or circuit: its name and members."""

    kind: str  # "superposition" or "circuit"
    name: str  # L3/S0123456789ab, or C4
    slug: str  # its registry slug, for a superposition; its id, for a circuit
    neurons: tuple[tuple[int, int, float], ...]  # (layer, neuron, weight)
    heads: tuple[tuple[int, int], ...] = ()  # (layer, head)
    definition: dict | None = field(default=None, compare=False, hash=False)  # a superposition's

    @property
    def units(self) -> list[tuple[int, int, str]]:
        """Every unit key of ``explain.attribute`` that the group covers."""
        poles = [(la, n, p) for la, n, _ in self.neurons for p in ("positive", "negative")]
        return poles + [(la, h, HEAD) for la, h in self.heads]


@dataclass
class GroupAttribution:
    """One superposition's or circuit's attribution, with what the registry says of it."""

    kind: str
    name: str
    members: int
    attribution: float
    status: str = "unstudied"
    explanation: str | None = None
    ablation_change: float | None = None


def superpositions(model: str, folder: Path | None = None) -> list[GroupDef]:
    """``model``'s superposition definitions, or none when it has no file."""
    from openinterpretability.core.identifiers import model_slug
    from openinterpretability.planner_superpositions import (
        DEFINITIONS_DIR,
        definition_identifier,
        load_definitions,
    )

    path = (folder or DEFINITIONS_DIR) / f"{model_slug(model)}.jsonl"
    if not path.exists():
        return []
    out = []
    for d in load_definitions(path):
        ident = definition_identifier(d)
        members = tuple((d["layer"], m["neuron_idx"], m["weight"]) for m in d["members"])
        name = ident.slug.split("/", 1)[1]
        out.append(GroupDef("superposition", name, ident.slug, members, definition=d))
    return out


def circuits(model: str, folder: Path | None = None) -> list[GroupDef]:
    """``model``'s current circuits, from ``data/circuits/``."""
    from openinterpretability import circuit_ids
    from openinterpretability.core.identifiers import model_slug
    from openinterpretability.harness.circuit_measures import Member

    path = circuit_ids.path_for(model_slug(model), folder or circuit_ids.CIRCUITS_DIR)
    if not path.exists():
        return []
    out = []
    for cid, hrefs in sorted(circuit_ids.load(path)[0].items()):
        members = [Member.parse(h) for h in sorted(hrefs)]
        neurons = tuple((m.layer, m.index, 1.0) for m in members if m.kind == "N")
        heads = tuple((m.layer, m.index) for m in members if m.kind == "H")
        out.append(GroupDef("circuit", cid, cid, neurons, heads))
    return out


def circuit_labels(wiki_dir: Path, model: str) -> dict[str, tuple[str, str | None]]:
    """``{circuit id: (status, explanation)}`` from each circuit's newest record."""
    from openinterpretability.core.circuit_records import CIRCUITS_WIKI, CircuitRecord
    from openinterpretability.core.identifiers import model_slug

    out: dict[str, tuple[str, str | None]] = {}
    root = wiki_dir / CIRCUITS_WIKI / model_slug(model)
    for folder in sorted(root.glob("C*")) if root.is_dir() else []:
        newest = max(folder.glob("*.json"), default=None)  # files are named by UTC time
        if newest is not None:
            record = CircuitRecord.model_validate_json(newest.read_text(encoding="utf-8"))
            out[folder.name] = (record.verdict.status, record.hypothesis.explanation)
    return out


def _direction(group: GroupDef):
    layer = group.neurons[0][0]
    index = torch.tensor([n for _, n, _ in group.neurons])
    weights = torch.tensor([w for *_, w in group.neurons], dtype=torch.float32)
    return layer, index, weights


def superposition_score(graph, group: GroupDef) -> float:
    """The direction's attribution (module docstring), from an ``explain.Graph``."""
    layer, index, weights = _direction(group)
    act, offset = graph.offset("N", layer)
    unit = weights / weights.norm()
    active = act[:, index] @ weights > 0
    along = (offset[:, index] @ unit) * (graph.grads[("N", layer)][:, index] @ unit)
    return float((along * active).sum())


def remove_direction(group: GroupDef, means):
    """A neuron intervention that removes the direction's offset where it is positive."""
    layer, index, weights = _direction(group)
    unit = weights / weights.norm()

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        part = act[..., 1:, :][..., index]  # the first token stays, as in the attribution
        mean = means.neurons[layer][index].to(act)
        u, w = unit.to(act), weights.to(act)
        along = ((part - mean) @ u) * (part @ w > 0)
        act[..., 1:, index] = part - along[..., None] * u
        return act

    return {layer: intervene}


def score(graph, scores: dict, group: GroupDef) -> float:
    """A group's attribution: a superposition's direction, or a circuit's members summed."""
    if group.kind == "superposition":
        return superposition_score(graph, group)
    return sum(scores.get(u, (0.0, 0.0))[0] for u in group.units)
