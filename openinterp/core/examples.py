"""What a scan shows of a unit.

A scan shows a unit's top-activating examples and the histogram of its strengths. For a
neuron, it shows the same for the other pole.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from openinterp.core.identifiers import Pole
from openinterp.core.strict import StrictModel, UnitFloat


class Histogram(StrictModel):
    """Counts of strengths in bins; ``bin_edges`` has one more entry than ``counts``."""

    bin_edges: list[float] = Field(min_length=2)
    counts: list[int]

    @model_validator(mode="after")
    def _shape(self) -> Histogram:
        if len(self.counts) != len(self.bin_edges) - 1:
            raise ValueError("counts must have exactly len(bin_edges) - 1 entries")
        return self

    def share_at_least(self, line: float) -> float:
        """Return the share of values at or above ``line``.

        Only bins wholly above ``line`` count, so the result is a lower bound, within one
        bin of the exact share.
        """
        total = sum(self.counts)
        above = sum(n for lo, n in zip(self.bin_edges, self.counts, strict=False) if lo >= line)
        return above / total if total else 0.0


class ActivatingExample(StrictModel):
    """A text snippet with per-token strengths (non-negative magnitudes).

    For an attention head, ``attends_to`` lists the tokens the peak token attends to
    most, strongest first, with their attention weights in ``attention_weights``.
    ``peak_labels`` gives the peak token's label in each of the scan's label sets (a
    document label, such as its language, is its document's); ``None`` when the scan had
    none or the peak token is unlabelled in all of them.
    """

    text: str
    tokens: list[str]
    activations: list[float]
    max_activation: float = Field(ge=0.0)
    max_token_index: int = Field(ge=0)
    attends_to: list[int] = Field(default_factory=list)
    attention_weights: list[UnitFloat] = Field(default_factory=list)
    peak_labels: dict[str, str] | None = None

    @model_validator(mode="after")
    def _aligned(self) -> ActivatingExample:
        if len(self.tokens) != len(self.activations):
            raise ValueError("tokens and activations must be the same length")
        if self.tokens and self.max_token_index >= len(self.tokens):
            raise ValueError("max_token_index out of range")
        if len(self.attends_to) != len(self.attention_weights):
            raise ValueError("attends_to and attention_weights must be the same length")
        if any(not 0 <= i <= self.max_token_index for i in self.attends_to):
            raise ValueError("attends_to must index tokens up to the peak (attention is causal)")
        return self


class OppositePole(StrictModel):
    """A neuron's other pole, read from the same scan as the audited pole.

    The neuron's page shows it beside the audited pole. It is reported, not gated, and
    neither agent sees it.
    """

    pole: Pole
    top_examples: list[ActivatingExample]
    activation_histogram: Histogram | None = None


def check_opposite_pole(model) -> None:
    """Reject an ``opposite_pole`` that the unit cannot have.

    Only a neuron pole has an opposite pole, and it must not be the audited pole.
    """
    if model.opposite_pole is None:
        return
    pole = model.identifier.pole
    if pole is None:
        raise ValueError("opposite_pole is only for neuron poles")
    if model.opposite_pole.pole == pole:
        raise ValueError("opposite_pole must be the pole the identifier does not name")
