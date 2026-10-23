"""Schemas for a Patchscope readout: the model describing a unit's output direction.

A Patchscope (Ghandeharioun et al., 2024) replaces the residual stream at one token of
an inspection prompt and lets the model continue. Here the replacement is the unit's
output direction alone (a pole's signed ``w_out``, or a dictionary latent's decoder
row), so the text is an example-free description of what the unit writes, in the
spirit of SelfIE (Chen et al., 2024). Reported, not gated (docs/PROTOCOL.md §4), and
shown to neither the proposer nor the reviewer.
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel


class PatchscopeReading(StrictModel):
    """One greedy continuation of one inspection prompt, with the direction injected."""

    prompt: str = Field(description="Name of the inspection prompt (a key of `prompts`).")
    layer: int = Field(
        ge=0, description="Layer after whose block the residual at the placeholder is replaced."
    )
    scale: float = Field(
        gt=0.0,
        description="Norm of the injected vector, as a multiple of the norm of the residual "
        "it replaces (the unpatched prompt's, at that layer and position).",
    )
    text: str = Field(description="The continuation, up to its first line break.")
    control_text: str = Field(
        description="The same with a random direction injected instead (`control_seed`): what "
        "the prompt says about a direction that means nothing in particular."
    )


class PatchscopeReadout(StrictModel):
    """Every reading of the unit's output direction, over prompts, layers and scales."""

    prompts: dict[str, str] = Field(
        description="Inspection prompts, with `{x}` marking the placeholder token's position."
    )
    placeholder: str = Field(description="Text of the token whose residual is replaced.")
    max_new_tokens: int = Field(gt=0)
    control_seed: int
    readings: list[PatchscopeReading]

    @property
    def scales(self) -> list[float]:
        return sorted({r.scale for r in self.readings})

    @property
    def rows(self) -> list[tuple[str, int, list[PatchscopeReading]]]:
        """(prompt, layer, its readings by ascending scale), in recorded order."""
        rows: dict[tuple[str, int], list[PatchscopeReading]] = {}
        for r in self.readings:
            rows.setdefault((r.prompt, r.layer), []).append(r)
        return [(p, layer, sorted(rs, key=lambda r: r.scale)) for (p, layer), rs in rows.items()]
