"""Identifiers of the units an experiment can audit.

The units are MLP neuron poles, attention heads and the latents of published sparse
dictionaries (transcoders, SAEs). A superposition is a unit too: a feature spread over
several of one layer's neurons (issue #3215).
"""

from __future__ import annotations

import re
from typing import Annotated, ClassVar, Literal

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
        """``Qwen--Qwen2.5-0.5B/L3/N7-positive``: the path of the unit's record and page."""
        return f"{self.model_slug}/L{self.layer}/N{self.neuron_idx}-{self.pole}"

    @property
    def index(self) -> int:
        """The neuron's index in its layer, under the name that every unit kind shares."""
        return self.neuron_idx

    noun: ClassVar[str] = "neuron"

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
        """``Qwen--Qwen2.5-0.5B/L3/H7``: the path of the unit's record and page."""
        return f"{self.model_slug}/L{self.layer}/H{self.head_idx}"

    @property
    def index(self) -> int:
        """The head's index in its layer, under the name that every unit kind shares."""
        return self.head_idx

    noun: ClassVar[str] = "attention head"

    @property
    def label(self) -> str:
        """``L3/H7``: the unit within its model."""
        return f"L{self.layer}/H{self.head_idx}"

    def __str__(self) -> str:
        return f"{self.model_name} {self.label}"


class FeatureIdentifier(_Component):
    """Addresses one latent of a published sparse dictionary (a transcoder or SAE) on an MLP.

    The learned-basis baseline for the native-neuron premise. Its experiments run under
    the same protocol, but never count towards native-neuron statistics. ``dictionary``
    names a preset in :mod:`openinterp.harness.dictionaries`. The submission's
    :class:`~openinterp.core.schemas.DictionaryProfile` pins the weights of the
    measurement (repository, commit, file). Latents are non-negative, so there is no pole.
    """

    kind: Literal["dictionary_feature"] = "dictionary_feature"
    dictionary: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", examples=["gpt2-small-transcoders"])
    feature_idx: int = Field(ge=0)

    @property
    def pole(self) -> None:
        """Latents have no sign; lets code read ``ident.pole`` for any unit."""
        return None

    @property
    def slug(self) -> str:
        """``<model>/L3/<dictionary>/F7``: the path of the unit's record and page."""
        return f"{self.model_slug}/L{self.layer}/{self.dictionary}/F{self.feature_idx}"

    @property
    def index(self) -> int:
        """The latent's index in its dictionary, under the name that every unit kind shares."""
        return self.feature_idx

    noun: ClassVar[str] = "dictionary feature"

    @property
    def label(self) -> str:
        """``L3/F7``: the unit within its model and dictionary."""
        return f"L{self.layer}/F{self.feature_idx}"

    def __str__(self) -> str:
        return f"{self.model_name} {self.label} ({self.dictionary})"


class SuperpositionIdentifier(_Component):
    """Addresses one superposition: a sparse direction over one layer's MLP neurons.

    Its members and weights are in the submission's
    :class:`~openinterp.core.profiles.SuperpositionProfile`. ``superposition_id`` is a
    hash of them (:func:`~openinterp.core.profiles.superposition_id`), so one direction
    always has one id. Strength is the positive part of the direction, so there is no pole.
    """

    kind: Literal["superposition"] = "superposition"
    superposition_id: str = Field(pattern=r"^[0-9a-f]{12}$")

    @property
    def pole(self) -> None:
        """A superposition has no sign; lets code read ``ident.pole`` for any unit."""
        return None

    @property
    def slug(self) -> str:
        """``Qwen--Qwen2.5-0.5B/L3/S0123456789ab``: the path of its record and page."""
        return f"{self.model_slug}/L{self.layer}/S{self.superposition_id}"

    @property
    def index(self) -> int:
        """The direction's latent in its in-memory dictionary, which is always the first."""
        return 0

    @property
    def feature_idx(self) -> int:
        """The same as :attr:`index`, under the name that the dictionary code reads."""
        return 0

    noun: ClassVar[str] = "superposition"

    @property
    def label(self) -> str:
        """``L3/S0123456789ab``: the unit within its model."""
        return f"L{self.layer}/S{self.superposition_id}"

    def __str__(self) -> str:
        return f"{self.model_name} {self.label}"


def _component_kind(value) -> str:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return kind or "mlp_neuron_pole"  # records predating `kind` are neuron poles


ComponentIdentifier = Annotated[
    Annotated[NeuronIdentifier, Tag("mlp_neuron_pole")]
    | Annotated[HeadIdentifier, Tag("attn_head")]
    | Annotated[FeatureIdentifier, Tag("dictionary_feature")]
    | Annotated[SuperpositionIdentifier, Tag("superposition")],
    Discriminator(_component_kind),
]
"""Any auditable unit: an MLP neuron pole, an attention head, a dictionary feature or a
superposition, told apart by ``kind``."""
