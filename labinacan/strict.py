"""The base of every schema model: strict, closed and frozen (#6628)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """The base of every schema.

    Strict types and ``extra="forbid"`` reject a record that does not match its schema.
    ``frozen`` stops code from changing a model after validation, because records are signed.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
