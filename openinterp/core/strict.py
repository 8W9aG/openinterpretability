"""Base types shared by the schema modules."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
