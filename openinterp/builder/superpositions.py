"""Superpositions on the site: their members, and the links from each member neuron.

A superposition is a sparse direction over one layer's MLP neurons
(:class:`~openinterp.core.profiles.SuperpositionProfile`). Its page lists its members. Each
member neuron's page lists the superpositions that it is a member of. A member neuron
with no experiment of its own gets a short page that lists them. Such a neuron counts
as studied, apart from the coverage numbers (:attr:`.tree.LayerNode.member_only`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from openinterp.builder.tree import SIDES, LayerNode, ModelNode, UnitNode, neuron_href

(SIDE,) = SIDES["superposition"]  # what a superposition's page groups its experiments under


@dataclass
class Member:
    """One member neuron of a superposition, as the superposition's page shows it."""

    neuron_idx: int
    weight: float
    relative: float  # the weight over the largest |weight| of the superposition
    href: str  # the member neuron's page


@dataclass
class Membership:
    """One superposition that a neuron is a member of, as the neuron's page shows it."""

    superposition: UnitNode
    relative: float  # this neuron's weight over the largest |weight| of the superposition

    @property
    def standing(self):
        """The superposition's standing."""
        return self.superposition.standings.get(SIDE)


def members_of(model: ModelNode, unit: UnitNode) -> list[Member]:
    """The members of the superposition ``unit``, from its newest record, by |weight|.

    It also keeps that record's profile on ``unit``, for its method and source.
    """
    newest = max(unit.experiments, key=lambda e: (e.record.verdict.timestamp, e.id))
    profile = unit.profile = newest.record.submission.superposition_profile
    if profile is None:
        return []
    scale = max(abs(m.weight) for m in profile.members)
    members = [
        Member(
            m.neuron_idx,
            m.weight,
            m.weight / scale,
            neuron_href(model.slug, unit.layer, m.neuron_idx),
        )
        for m in profile.members
    ]
    return sorted(members, key=lambda m: (-abs(m.weight), m.neuron_idx))


def add_members(models: Sequence[ModelNode]) -> None:
    """Give each superposition its members, and link each member neuron back to it.

    A member neuron with no experiment of its own gets a unit with no experiments in its
    layer's ``members``. The site renders that unit as a short page.
    """
    for model in models:
        for layer in model.layers:
            for unit in layer.superpositions:
                unit.members = members_of(model, unit)
                for m in unit.members:
                    neuron = _neuron(model, layer, m)
                    neuron.part_of.append(Membership(unit, m.relative))
            for neuron in [*layer.neurons, *layer.members]:
                neuron.part_of.sort(key=lambda x: (-abs(x.relative), x.superposition.label))


def _neuron(model: ModelNode, layer: LayerNode, member: Member) -> UnitNode:
    """The member's neuron unit: its studied unit, else its page with no experiment."""
    for neuron in [*layer.neurons, *layer.members]:
        if neuron.idx == member.neuron_idx:
            return neuron
    neuron = UnitNode(model.name, layer.layer, member.neuron_idx, member.href)
    layer.members.append(neuron)
    layer.members.sort(key=lambda n: n.idx)
    return neuron


def count_pageless_neurons(totals: dict, models: Sequence[ModelNode]) -> None:
    """Add the studied neurons with no experiment to the home page's ``totals``.

    The registry query counts only neurons with an experiment. This adds the neurons whose
    scans all stopped, and the superposition members with no experiment or stopped scan.
    ``totals["member_only"]`` is the number of those members.
    """
    layers = [layer for m in models for layer in m.layers]
    totals["member_only"] = sum(layer.member_only for layer in layers)
    totals["neurons"] += totals["member_only"] + sum(len(layer.stopped) for layer in layers)
