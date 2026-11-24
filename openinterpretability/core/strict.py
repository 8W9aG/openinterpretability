"""Base types shared by the schema modules. ``StrictModel`` lives in labinacan (#6628)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from labinacan.strict import StrictModel

__all__ = ["StrictModel", "UnitFloat"]

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
