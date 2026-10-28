"""A neuron pole's strongest upstream and downstream connections (reported, not gated)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.strict import StrictModel

WiringMethod = Literal["weights", "activations"]


class WiringEdge(StrictModel):
    """One connection between the audited pole and another unit of the same model.

    ``kind`` and ``layer``/``index`` name the other unit: a neuron (either pole) or an
    attention head. What ``weight`` measures depends on :attr:`Wiring.method`.
    ``channels`` breaks a weights reading down by the read it goes through: ``in``, or
    ``gate`` and ``up`` for a gated MLP; ``q``, ``k`` and ``v`` for a head downstream.
    ``weight`` is then the channel of largest magnitude.
    """

    kind: Literal["mlp_neuron", "attn_head"]
    layer: int = Field(ge=0)
    index: int = Field(ge=0)
    weight: float
    channels: dict[str, float] = Field(default_factory=dict)


class Wiring(StrictModel):
    """A unit's top connections, from its weights or from activations.

    A neuron pole reads its MLP input (``gate``/``up``, or ``in``) and writes ``w_out``
    (negated for the negative pole); a dictionary feature reads with its encoder row and
    writes its decoder row; a head reads through ``q``, ``k`` and ``v`` and writes the
    subspace ``W_O``. Upstream units write before the unit reads (a head's own layer's
    MLP runs after it, and reads it downstream).

    *weights* (virtual weights, Elhage et al., 2021). The norms before each read are
    folded in as their gain (and centring, for LayerNorm); their per-token scale is
    ignored, so weights rank units but have no absolute meaning. Each number is how much
    one unit's read picks up of the other's write: ``w · r̂`` when a vector reads a
    vector (``+`` means a positive activation raises the read; the unit's own reads are
    unit-normalised upstream), else a norm (≥ 0): of ``R·w`` for a matrix read, of
    ``R·W_O·W_V`` for an upstream head's OV circuit, of ``W_Q/K/V·w`` downstream.

    *activations* (attribution patching, Syed et al., 2023), averaged over the peaks of
    the review's prompts where the unit fired:

    * upstream: ``Σ_s write_s · ∂strength/∂x_s``, the first-order change in strength from
      removing an earlier unit's direct write to its input (``+`` means it excites);
    * downstream neurons: the exact change in a later neuron's signed activation at the
      peak when the unit's write is removed from its input alone;
    * downstream heads: how far a later head's write at the peak moves, ``|W_O Δz|``,
      when the unit's write is removed from its ``q``, ``k`` or ``v`` input alone, every
      position at once (one channel at a time).
    """

    method: WiringMethod
    upstream_neurons: list[WiringEdge]
    upstream_heads: list[WiringEdge] | None = Field(
        default=None, description="None when the architecture's attention isn't supported."
    )
    downstream_neurons: list[WiringEdge]
    downstream_heads: list[WiringEdge] | None = Field(
        default=None,
        description="None when not measured (activations before downstream heads were).",
    )
    prompts: int = Field(
        default=0, ge=0, description="Activations only: prompts whose peaks were averaged."
    )
