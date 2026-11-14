"""The draw ledger: every draw, recorded before its outcome is known (issue #3205).

``wiki/draws/`` keeps only the random draws that the scan stopped on. A draw whose run was
dropped, or crashed, left no file. Rates over random draws then counted only the runs that
finished. So ``open-interpretability next --steps`` writes a :class:`DrawEntry` when it makes the
draw, and the claim commit adds it. A run that stops lands its PR with only that entry.

An entry's outcome is read later, from the registry (:func:`resolve`):

- ``record``: a record of the unit, from the entry's seed or at or after its time.
- ``stopped``: a file in ``wiki/draws/`` or ``wiki/stops/`` for the unit.
- ``none``: no outcome yet. The run was dropped, or it is still running.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime

from openinterp import __version__
from openinterp.core.paths import ledger_dir
from openinterp.core.strict import StrictModel

Outcome = Literal["record", "stopped", "none"]


class DrawEntry(StrictModel):
    """One draw of ``open-interpretability next``, as it stood before the run."""

    timestamp: AwareDatetime
    seed: int  # the planner's seed: `open-interpretability next --seed` draws it again
    kind: str  # explore, revise, independent_review, …
    model: str
    slug: str | None = None  # None when the unit is known only once the scan starts
    record: str | None = None  # the record that a follow-up builds on
    audit: str | None = None  # the `open-interpretability audit` options of a draw that scans
    pole_forced: bool | None = None  # for a random draw: --pole fixed the pole
    harness_version: str = __version__

    @property
    def audit_seed(self) -> int | None:
        """The ``--seed`` that a random draw's audit passes, if any."""
        options = (self.audit or "").split()
        if "--random" in options and "--seed" in options:
            return int(options[options.index("--seed") + 1])
        return None


def ledger_path(wiki_dir: Path, entry: DrawEntry) -> Path:
    """Where ``entry`` is stored: ``ledger/<YYYYMMDD>/<timestamp>-<seed>.json``."""
    stamp = entry.timestamp.strftime("%Y%m%dT%H%M%SZ")
    return ledger_dir(wiki_dir) / stamp[:8] / f"{stamp}-{entry.seed}.json"


def write_entry(entry: DrawEntry, wiki_dir: Path) -> Path:
    """Store ``entry`` (write-once: an existing file is left as it is)."""
    path = ledger_path(wiki_dir, entry)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.model_dump_json() + "\n", encoding="utf-8")
    return path


def ledger_files(wiki_dir: Path) -> Iterator[Path]:
    """Every ledger file under ``wiki_dir``, sorted by path; none without a ledger."""
    root = ledger_dir(wiki_dir)
    yield from sorted(root.rglob("*.json")) if root.is_dir() else ()


def load_ledger(wiki_dir: Path) -> list[DrawEntry]:
    """Every ledger entry, oldest first."""
    entries = [DrawEntry.model_validate_json(p.read_text()) for p in ledger_files(wiki_dir)]
    return sorted(entries, key=lambda e: e.timestamp)


Row = tuple[str, int | None, datetime, str | None]  # slug, random seed, time, status


def resolve(entry: DrawEntry, records: list[Row], stopped: list[Row]) -> tuple[Outcome, str | None]:
    """What became of ``entry``, and the status of its first matching record, if any.

    A random draw matches by its audit seed and model. Any draw also matches its slug at
    or after its time. ``records`` and ``stopped`` hold (slug, random seed, time, status).
    """

    def first_match(rows: list[Row]) -> Row | None:
        seed, prefix = entry.audit_seed, _model_slug(entry)
        hits = [
            row
            for row in rows
            if (seed is not None and row[1] == seed and row[0].startswith(prefix))
            or (entry.slug is not None and row[0] == entry.slug and row[2] >= entry.timestamp)
        ]
        return min(hits, key=lambda row: row[2]) if hits else None

    if (row := first_match(records)) is not None:
        return "record", row[3]
    if first_match(stopped) is not None:
        return "stopped", None
    return "none", None


def summary(resolved: list[tuple[DrawEntry, Outcome, str | None]]) -> dict:
    """Counts by outcome, and the VERIFIED share of explore draws: of all, and of finished ones.

    The first share counts a draw with no outcome as not VERIFIED. The second leaves it
    out. So the two bracket the share that the dropped runs would have given.
    """
    explore = [(o, s) for e, o, s in resolved if e.kind == "explore"]
    finished = [s for o, s in explore if o == "record"]
    verified = sum(s == "VERIFIED" for s in finished)
    return {
        "entries": len(resolved),
        "outcomes": {o: sum(r[1] == o for r in resolved) for o in ("record", "stopped", "none")},
        "explore_entries": len(explore),
        "explore_verified": verified,
        "verified_share_of_all_explore": verified / len(explore) if explore else None,
        "verified_share_of_finished_explore": verified / len(finished) if finished else None,
        "pole_forced": sum(bool(e.pole_forced) for e, _, _ in resolved),
    }


def _model_slug(entry: DrawEntry) -> str:
    from openinterp.core.identifiers import model_slug

    return model_slug(entry.model) + "/"
