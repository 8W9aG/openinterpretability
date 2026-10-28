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
    """A neuron pole's top connections, from its weights or from activations.

    *weights* (virtual weights, Elhage et al., 2021). The norms before each read are
    folded in as their gain (and centring, for LayerNorm); their per-token scale is
    ignored, so weights rank units but have no absolute meaning.

    * upstream neurons: ``w_out · r̂``, an earlier neuron's write against the pole's
      unit-normalised read ``r̂`` (each of ``gate``/``up``, or ``in``), signed against
      those weights rather than the pole: ``+`` means its positive activation raises the
      pre-activation;
    * upstream heads (this layer and earlier): ``|r̂ᵀ W_O W_V|``, the norm of the OV
      circuit's read of the pole's input, so ≥ 0;
    * downstream neurons: ``r · w``, a later neuron's read against the pole's write
      (``w_out``, negated for the negative pole);
    * downstream heads: ``|W_Q w|``, ``|W_K w|``, ``|W_V w|``, so ≥ 0.

    *activations* (attribution patching, Syed et al., 2023), averaged over the peaks of
    the reviewer's prompts where the pole fired:

    * upstream: ``write · ∇strength``, the first-order change in pole strength from
      removing an earlier unit's direct write to its input (``+`` means it excites the
      pole);
    * downstream neurons: the exact change in a later neuron's signed activation when
      the pole's write is removed from its input alone, everything else held fixed.
      Downstream heads are not measured (``None``).
    """

    method: WiringMethod
    upstream_neurons: list[WiringEdge]
    upstream_heads: list[WiringEdge] | None = Field(
        default=None, description="None when the architecture's attention isn't supported."
    )
    downstream_neurons: list[WiringEdge]
    downstream_heads: list[WiringEdge] | None = None
    prompts: int = Field(
        default=0, ge=0, description="Activations only: prompts whose peaks were averaged."
    )
