"""Explanation records: why a model gives one answer, as the registry stood (issue #3744).

``open-interpretability explain --record`` writes one. It holds the prompt and the answer,
the shares and prompt coverage. It holds the mechanism: each unit and group, with its share
and status. It also holds the narrated long form and paragraph (#3743), with the narrator.
Each record lives at ``wiki/explanations/<model slug>/<prompt id>/<UTC>.json``, so a prompt
keeps its history as the registry grows.

An explanation is a snapshot. Each unit's status is the one at the time of writing. When a
cited unit's status has changed since, the site marks the explanation stale
(:mod:`openinterpretability.builder.explanation_pages`), and a newer record replaces it.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import Field

from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.strict import StrictModel

EXPLANATIONS_WIKI = "explanations"  # records live in wiki/explanations/


class ExplainedUnit(StrictModel):
    """One unit or group of the mechanism, as it stood when the explanation was written."""

    unit: str = Field(description="L3/N7+, L2/H10, L20/S0123456789ab or C4.")
    slug: str | None = Field(default=None, description="Its registry slug, for a pole or head.")
    kind: str  # pole, head, superposition or circuit
    share: float
    attribution: float
    status: str
    explanation: str | None = None  # only when VERIFIED


class Narration(StrictModel):
    """The narrated levels (#3743)."""

    sections: list[dict[str, str]] = Field(default_factory=list)  # {title, text}
    paragraph: str
    narrator_model: str
    citations_not_in_evidence: list[str] = Field(default_factory=list)


class ExplanationRecord(StrictModel):
    """One explanation of one answer, as the registry stood (module docstring)."""

    model_name: str
    prompt: str
    position: int
    answer: str
    explained_share: float
    blocked_share: float
    prompt_coverage: float | None
    random_coverage: float | None
    units: list[ExplainedUnit]
    narration: Narration | None = None
    planned_experiments: int = 0  # unit experiments that a complete explanation needs
    package_version: str
    written: datetime

    @property
    def prompt_id(self) -> str:
        """The prompt's id: the start of the SHA-256 of the model, position and prompt."""
        key = f"{self.model_name}\n{self.position}\n{self.prompt}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]


def record_path(wiki_dir: Path, record: ExplanationRecord) -> Path:
    """``wiki/explanations/<model slug>/<prompt id>/<UTC timestamp>.json``."""
    folder = wiki_dir / EXPLANATIONS_WIKI / model_slug(record.model_name) / record.prompt_id
    return folder / f"{record.written:%Y%m%dT%H%M%SZ}.json"


def load_latest(wiki_dir: Path) -> list[tuple[Path, ExplanationRecord]]:
    """The newest record of each prompt, by model and prompt id; none without the folder."""
    root = wiki_dir / EXPLANATIONS_WIKI
    out = []
    for folder in sorted(p for p in root.glob("*/*") if p.is_dir()) if root.is_dir() else []:
        newest = max(folder.glob("*.json"), default=None)  # files are named by UTC time
        if newest is not None:
            out.append((newest, ExplanationRecord.model_validate_json(newest.read_text())))
    return out
