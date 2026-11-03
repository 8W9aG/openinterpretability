"""Base types shared by the schema modules."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """The base of every schema.

    Strict types and ``extra="forbid"`` reject a record that does not match its schema.
    ``frozen`` stops code from changing a model after validation, because records are signed.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
