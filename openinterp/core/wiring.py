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
    ``channels`` splits a weights reading by the read that it goes through. The reads
    are ``in``, or ``gate`` and ``up`` for a gated MLP, or ``q``, ``k`` and ``v`` for a
    head downstream. ``weight`` is then the channel of largest magnitude.
    """

    kind: Literal["mlp_neuron", "attn_head"]
    layer: int = Field(ge=0)
    index: int = Field(ge=0)
    weight: float
    channels: dict[str, float] = Field(default_factory=dict)


class Wiring(StrictModel):
    """A unit's top connections, from its weights or from activations.

    How each kind of unit reads and writes:

    - A neuron pole reads its MLP input (``gate``/``up``, or ``in``). It writes ``w_out``,
      negated for the negative pole.
    - A dictionary feature reads with its encoder row and writes its decoder row.
    - A head reads through ``q``, ``k`` and ``v``, and writes the subspace ``W_O``.

    Upstream units write before the unit reads. A head's own layer's MLP runs after the
    head, so it reads the head downstream.

    *weights* (virtual weights, Elhage et al., 2021). The norms before each read are
    folded in as their gain, and centring for LayerNorm. Their per-token scale is
    ignored. So weights rank units, but have no absolute meaning. Each number is how much
    one unit's read picks up of the other unit's write:

    - When a vector reads a vector: ``w · r̂``. ``+`` means that a positive activation
      raises the read. Upstream, the unit's own reads are unit-normalised.
    - Otherwise, a norm (≥ 0). For a matrix read, it is the norm of ``R·w``. For an
      upstream head's OV circuit, it is the norm of ``R·W_O·W_V``. Downstream, it is the
      norm of ``W_Q/K/V·w``.

    *activations* (attribution patching, Syed et al., 2023). Each number is averaged over
    the peaks of the review's prompts where the unit fired:

    - Upstream: ``Σ_s write_s · ∂strength/∂x_s``. This is the first-order change in
      strength when the earlier unit's direct write is removed from the input. ``+``
      means that the earlier unit excites the unit.
    - Downstream neurons: the exact change in a later neuron's signed activation at the
      peak. The unit's write is removed from that neuron's input only.
    - Downstream heads: ``|W_O Δz|``, how far a later head's write at the peak moves. The
      harness removes the unit's write from one input of the head at a time (``q``,
      ``k`` or ``v``). It removes the write at every position at once.
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
