"""Identifiers of the units an experiment can audit: MLP neuron poles and attention heads."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Discriminator, Field, Tag

from openinterp.core.strict import StrictModel

Pole = Literal["positive", "negative"]


def model_slug(model_name: str) -> str:
    """Path-safe model name: ``Qwen/Qwen2.5-0.5B`` → ``Qwen--Qwen2.5-0.5B``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_name)


class _Component(StrictModel):
    model_name: str = Field(min_length=1, examples=["Qwen/Qwen2.5-0.5B"])
    layer: int = Field(ge=0)

    @property
    def model_slug(self) -> str:
        return model_slug(self.model_name)


class NeuronIdentifier(_Component):
    """Addresses one pole of one MLP neuron in an open-weight model.

    The ``mlp_neuron_pole`` case of :data:`ComponentIdentifier`. ``kind`` defaults, so
    records written before it existed still validate.
    """

    kind: Literal["mlp_neuron_pole"] = "mlp_neuron_pole"
    neuron_idx: int = Field(ge=0)
    pole: Pole

    @property
    def slug(self) -> str:
        return f"{self.model_slug}/L{self.layer}/N{self.neuron_idx}-{self.pole}"

    @property
    def index(self) -> int:
        return self.neuron_idx

    @property
    def label(self) -> str:
        """``L3/N7+``: the unit within its model."""
        return f"L{self.layer}/N{self.neuron_idx}{'+' if self.pole == 'positive' else '-'}"

    def __str__(self) -> str:
        return f"{self.model_name} {self.label}"


class HeadIdentifier(_Component):
    """Addresses one attention head: its slice of the attention output projection's input.

    Heads have no sign, so no pole. Their per-token strength is defined by
    :class:`HeadProfile` (docs/PROTOCOL.md §1).
    """

    kind: Literal["attn_head"] = "attn_head"
    head_idx: int = Field(ge=0)

    @property
    def pole(self) -> None:
        """Heads have no sign; lets code read ``ident.pole`` for any unit."""
        return None

    @property
    def slug(self) -> str:
        return f"{self.model_slug}/L{self.layer}/H{self.head_idx}"

    @property
    def index(self) -> int:
        return self.head_idx

    @property
    def label(self) -> str:
        return f"L{self.layer}/H{self.head_idx}"

    def __str__(self) -> str:
        return f"{self.model_name} {self.label}"


def _component_kind(value) -> str:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return kind or "mlp_neuron_pole"  # records predating `kind` are neuron poles


ComponentIdentifier = Annotated[
    Annotated[NeuronIdentifier, Tag("mlp_neuron_pole")]
    | Annotated[HeadIdentifier, Tag("attn_head")],
    Discriminator(_component_kind),
]
"""Any auditable unit: an MLP neuron pole or an attention head, told apart by ``kind``."""
