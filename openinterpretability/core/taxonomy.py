"""The registry's taxonomy: what each kind of thing is, and how the kinds nest (#3703).

This module is the one definition of each concept. The About page reads it, and
``scripts/taxonomy_table.py`` writes it into ``docs/PROTOCOL.md`` §1 as a table. A test
fails when the table and this module differ. The precise definitions of the audited units
(the activations, the strength, the identifiers' fields) stay in the unit table of §1.

The identifier forms are examples. A test builds each one from its identifier class.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Values of :attr:`Concept.audited`.
AUDITED = "yes"
NOT_AUDITED = "no"


@dataclass(frozen=True)
class Concept:
    """One kind of thing in the registry."""

    term: str
    definition: str
    part_of: str  # what it belongs to, or "" for a model
    contains: str  # what it holds, or ""
    audited: str  # AUDITED when it has its own experiments, verdicts and page
    identifier: str  # an example of its identifier form
    sister: str  # what its sister in another model is, and whether the registry has one

    @property
    def anchor(self) -> str:
        """The id of the concept's row on the About page."""
        return "taxonomy-" + self.term.lower().replace(" ", "-")


TAXONOMY: tuple[Concept, ...] = (
    Concept(
        "Model",
        "An open-weight language model at a pinned revision.",
        "",
        "layers",
        NOT_AUDITED,
        "openai-community/gpt2",
        "Planned: models correspond through their layers, by relative depth (#4441).",
    ),
    Concept(
        "Layer",
        "One transformer block: an attention sublayer, then an MLP sublayer.",
        "a model",
        "neurons, heads, superpositions and dictionary features",
        NOT_AUDITED,
        "L3",
        "Planned: the layer at the same relative depth in another model (#4441).",
    ),
    Concept(
        "Neuron",
        "One coordinate of a layer's MLP hidden activation. Its page shows both of its poles.",
        "a layer",
        "two poles",
        "through its poles",
        "L3/N7",
        "Through its poles.",
    ),
    Concept(
        "Pole",
        "The positive or the negative half of a neuron's activation. "
        "Most experiments audit one neuron pole.",
        "a neuron",
        "",
        AUDITED,
        "L3/N7+",
        "In the registry: a mutual best match by activation over a shared corpus "
        "(`open-interpretability sisters`).",
    ),
    Concept(
        "Head",
        "One attention head of a layer. A head has no sign, so it has no poles.",
        "a layer",
        "",
        AUDITED,
        "L5/H1",
        "Planned: a head in another model whose behaviour matches (#4438).",
    ),
    Concept(
        "Superposition",
        "A feature that one layer spreads over several of its neurons. "
        "It is a direction within one layer, never across layers.",
        "a layer",
        "member neurons, each with a weight",
        AUDITED,
        "L2/S0123456789ab",
        "Planned: a superposition, or a pole, in another model with the same feature (#4439).",
    ),
    Concept(
        "Dictionary feature",
        "A latent of a published sparse dictionary (an SAE or a transcoder) on one layer. "
        "It is a learned direction, the baseline for native neurons, and never counted with them.",
        "a layer",
        "",
        AUDITED,
        "L4/F12",
        "None planned: it is a baseline.",
    ),
    Concept(
        "Circuit",
        "Units in several layers that work together: a cluster of the wiring graph between "
        "studied units. A task circuit traced by attribution is an analysis, not a registry "
        "circuit.",
        "a model",
        "neurons and heads from several layers",
        AUDITED,
        "C3",
        "Planned: a circuit in another model that does the same task (#4440).",
    ),
    Concept(
        "Hub",
        "A neuron or head that the wiring joins to many studied units. "
        "Circuits leave hubs out, and a hub has no experiments of its own.",
        "a model",
        "its center, a neuron or a head",
        NOT_AUDITED,
        "L3/N7/HUB",
        "Through its center.",
    ),
    Concept(
        "Experiment",
        "One test of one hypothesis about one audited unit, with a verdict.",
        "an audited unit",
        "a hypothesis, probes and a verdict",
        NOT_AUDITED,
        "20261019T050928Z",
        "A sister study tests a pole's explanation on its sister.",
    ),
    Concept(
        "Record",
        "The signed, write-once file of one experiment, in the wiki.",
        "an experiment",
        "the submission, the review and the provenance",
        NOT_AUDITED,
        "wiki/experiments/<unit>/<time>.json",
        "",
    ),
)


def protocol_table() -> str:
    """The taxonomy as the Markdown table of ``docs/PROTOCOL.md`` §1."""
    head = "| Term | Definition | Part of | Audited | Identifier | Sister |\n"
    rule = "| --- | --- | --- | --- | --- | --- |\n"
    rows = "".join(
        f"| {c.term} | {c.definition} | {c.part_of or '—'} | {c.audited} | "
        f"`{c.identifier}` | {c.sister or '—'} |\n"
        for c in TAXONOMY
    )
    return head + rule + rows
