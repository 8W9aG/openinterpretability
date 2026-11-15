"""Random draws that could not be audited, kept as outcomes of random selection.

A random draw is one of the registry's unbiased samples. Statistics over random units
(share VERIFIED, share dead) need every draw, not only the draws that could be audited.
Two kinds of unit cannot be audited:

- *dead*: a unit that never fires on the scan. For a neuron pole, the scan then checks
  the weights (``openinterp.harness.dead``). When they prove that no input can make the
  pole fire, the draw is ``proven_dead``, and the planner counts the pole as verified.
- *default-on* (protocol 0.8): a unit active on at least half its tokens.

The scan stops on both. For a random draw, it writes an :class:`UnauditableDraw` instead
of evidence, once only, at ``wiki/draws/<slug>/<timestamp>.json`` (``open-interpretability audit
--random``).

For a chosen unit, such as a revise, it writes a :class:`ChosenStop` at
``wiki/stops/<slug>/<timestamp>.json``. A chosen unit is not a sample, and it can already
count as a random draw. So stops have their own directory, and :func:`load_draws` does not
read them. A proof by the weights does not depend on selection, so a ``proven_dead`` stop
also counts its pole as verified (:func:`openinterp.planner.dead`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime

from openinterp.core.paths import draws_dir, stops_dir
from openinterp.core.schemas import (
    PROTOCOL_VERSION,
    ComponentIdentifier,
    ExperimentProvenance,
)
from openinterp.core.strict import StrictModel, UnitFloat

DrawOutcome = Literal["dead", "default_on"]


class UnauditableDraw(StrictModel):
    """One random draw that the scan stopped on.

    It is kept because rates over random units are biased without every draw.
    """

    identifier: ComponentIdentifier
    outcome: DrawOutcome
    active_share: UnitFloat | None = None  # default-on: the share the scan stopped on
    proven_dead: bool = False  # dead: the weights prove that no input makes it fire
    dead_bound: float | None = None  # dead: the largest pre-activation any input gives
    provenance: ExperimentProvenance
    protocol_version: str = PROTOCOL_VERSION
    timestamp: AwareDatetime


class ChosenStop(UnauditableDraw):
    """One chosen unit that the scan stopped on.

    It is kept so that the registry shows the attempt. No random-selection statistic reads it.
    """

    latest_record: str | None = None  # the unit's newest experiment when the scan stopped


def draw_path(wiki_dir: Path, draw: UnauditableDraw) -> Path:
    """Where ``draw`` is stored: ``draws/<slug>/<timestamp>.json`` under ``wiki_dir``."""
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
    """Every draw file under ``wiki_dir``, sorted by path; none without a draws directory."""
    root = draws_dir(wiki_dir)
    yield from sorted(root.rglob("*.json")) if root.is_dir() else ()


def load_draws(wiki_dir: Path) -> list[UnauditableDraw]:
    """Every recorded draw, oldest first."""
    draws = [UnauditableDraw.model_validate_json(p.read_text()) for p in draw_files(wiki_dir)]
    return sorted(draws, key=lambda d: d.timestamp)


def stop_path(wiki_dir: Path, stop: ChosenStop) -> Path:
    """Where ``stop`` is stored: ``stops/<slug>/<timestamp>.json`` under ``wiki_dir``."""
    stamp = stop.timestamp.strftime("%Y%m%dT%H%M%SZ")
    return stops_dir(wiki_dir) / stop.identifier.slug / f"{stamp}.json"


def stop_files(wiki_dir: Path) -> Iterator[Path]:
    """Every stop file under ``wiki_dir``, sorted by path; none without a stops directory."""
    root = stops_dir(wiki_dir)
    yield from sorted(root.rglob("*.json")) if root.is_dir() else ()


def load_stops(wiki_dir: Path) -> list[ChosenStop]:
    """Every recorded stop of a chosen unit, oldest first."""
    stops = [ChosenStop.model_validate_json(p.read_text()) for p in stop_files(wiki_dir)]
    return sorted(stops, key=lambda s: s.timestamp)
