"""Schema for the terms of a model that the registry studies."""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel


class ModelLicense(StrictModel):
    """The terms of one model, from ``[tool.openinterp.licenses]`` in ``pyproject.toml``."""

    license: str = Field(min_length=1, description="The licence on the model card.")
    url: str = Field(description="Where the licence text is.")
    gated: bool = Field(description="True: the Hub asks users to accept terms before download.")
    publishing_allowed: bool = Field(
        description="True: the terms let the registry publish activations and generated text."
    )
    conditions: str | None = Field(
        default=None, description="What the terms ask of what the registry publishes."
    )
