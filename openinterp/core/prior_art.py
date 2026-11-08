"""Schema for prior art: what others have published about a unit, found after the proposer answers.

Today the only source is Neuronpedia. A dictionary latent that Neuronpedia hosts has its
own explanations there. A neuron pole has none, so the explanations of the published
latents whose weights align with it stand in. Reported, not gated (docs/PROTOCOL.md §4).
The proposer never sees it. The reviewer does, as a source of probes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from openinterp.core.strict import StrictModel


class PriorArt(StrictModel):
    """One published explanation of the unit, or of a latent aligned with it."""

    source: Literal["neuronpedia"] = "neuronpedia"
    dictionary: str = Field(description="The dictionary preset the latent belongs to.")
    feature_idx: int = Field(ge=0)
    relation: Literal["self", "writes", "reads"] = Field(
        description="self: the unit is this latent. writes or reads: the latent's decoder "
        "or encoder aligns with the pole's write or read."
    )
    cosine: float | None = Field(
        default=None, description="Signed alignment with the pole; None for self."
    )
    description: str = Field(min_length=1, description="The explanation, word for word.")
    explainer: str = Field(description="The model that wrote it, or 'human'.")
    url: str = Field(description="The latent's page at the source.")


# The submission's field (``NeuronAuditSubmission.prior_art``).
PriorArtField = Annotated[
    list[PriorArt] | None,
    Field(
        description="Published explanations of the unit, or of latents aligned with it, "
        "found after the proposer answered and shown to the reviewer only (reported, not "
        "gated). None: not looked up; empty: none found."
    ),
]
