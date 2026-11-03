"""Schemas for steering samples: what the model writes with a unit clamped high or ablated.

The harness continues each neutral prompt greedily in three ways: at baseline, with the
unit ablated, and with the unit clamped at every position. The clamp uses several
multiples of the unit's own top activation. You can then read the continuations side
by side (Turner et al., 2023; Goodfire, "Neural Networks Are Already Clean", 2026).
Reported, not gated (docs/PROTOCOL.md §4). Neither the proposer nor the reviewer sees
them.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from openinterp.core.strict import StrictModel


class SteeringSample(StrictModel):
    """One prompt's greedy continuations under each intervention."""

    prompt: str = Field(min_length=1)
    baseline: str = Field(description="The continuation with no intervention.")
    ablated: str = Field(description="The same with the unit ablated at every position.")
    clamped: list[str] = Field(
        description="The same with the unit clamped at every position, one per `multiples`."
    )


class SteeringSamples(StrictModel):
    """Every prompt's continuations, with the unit clamped high or ablated."""

    clamp_unit: float = Field(
        gt=0.0,
        description="The strength one multiple stands for: the unit's top activation in the "
        "scan (`max_activation`). A neuron pole is clamped to that signed value.",
    )
    multiples: list[float] = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)
    samples: list[SteeringSample]

    @model_validator(mode="after")
    def _one_text_per_multiple(self) -> SteeringSamples:
        for s in self.samples:
            if len(s.clamped) != len(self.multiples):
                raise ValueError(
                    f"{s.prompt!r} has {len(s.clamped)} clamped texts for "
                    f"{len(self.multiples)} multiples"
                )
        return self
