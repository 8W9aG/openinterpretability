"""Random draws that could not be audited, kept as outcomes of random selection.

A random draw is one of the registry's unbiased samples, so statistics over random units
(share VERIFIED, share dead) need every draw, not only the ones that could be audited.
Two kinds can't be: a unit that never fires on the scan (*dead*), and one active on at
least half its tokens (*default-on*, protocol 0.8). The scan stops on both and writes an
:class:`UnauditableDraw` instead of evidence, write-once, at
``wiki/draws/<slug>/<timestamp>.json`` (``openinterp audit --random``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime

from openinterp.core.paths import draws_dir
from openinterp.core.schemas import (
    PROTOCOL_VERSION,
    ComponentIdentifier,
    ExperimentProvenance,
)
from openinterp.core.strict import StrictModel, UnitFloat

DrawOutcome = Literal["dead", "default_on"]


class UnauditableDraw(StrictModel):
    identifier: ComponentIdentifier
    outcome: DrawOutcome
    active_share: UnitFloat | None = None  # default-on: the share the scan stopped on
    provenance: ExperimentProvenance
    protocol_version: str = PROTOCOL_VERSION
    timestamp: AwareDatetime


def draw_path(wiki_dir: Path, draw: UnauditableDraw) -> Path:
    stamp = draw.timestamp.strftime("%Y%m%dT%H%M%SZ")
    return draws_dir(wiki_dir) / draw.identifier.slug / f"{stamp}.json"


def write_draw(draw: UnauditableDraw, wiki_dir: Path) -> Path:
    """Store ``draw`` (write-once: an existing file is left as it is)."""
    path = draw_path(wiki_dir, draw)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(draw.model_dump_json() + "\n", encoding="utf-8")
    return path


def draw_files(wiki_dir: Path) -> Iterator[Path]:
    root = draws_dir(wiki_dir)
    yield from sorted(root.rglob("*.json")) if root.is_dir() else ()


def load_draws(wiki_dir: Path) -> list[UnauditableDraw]:
    """Every recorded draw, oldest first."""
    draws = [UnauditableDraw.model_validate_json(p.read_text()) for p in draw_files(wiki_dir)]
    return sorted(draws, key=lambda d: d.timestamp)
