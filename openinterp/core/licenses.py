"""The terms of the models that the registry studies, and of the data it publishes."""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel

# The licence of the registry's data: the records and pages under ``wiki/`` and the site
# built from them. The code is under ``LICENSE`` (MIT). Corpus excerpts inside records
# keep their corpus's own terms, which ``<name>.source.json`` records.
DATA_LICENSE = "CC-BY-4.0"
DATA_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATA_TERMS = {"data_license": DATA_LICENSE, "data_license_url": DATA_LICENSE_URL}  # for pages


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
