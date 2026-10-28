"""Wiring for archived records that predate it (`openinterp wiring-backfill`).

Archived records are write-once and many are signed, so their wiring is not written into
them. It goes to ``docs/analyses/wiring.jsonl``, one row per experiment, and the site
builder shows it for any record that has no wiring of its own. Both readings are taken
the way an audit takes them: the weights from the unit alone, and activations on the
prompts the review should have fired on (the counterfactuals' base prompts and the
positive holdouts), at the record's activation threshold, model revision and dtype.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from openinterp.core.schemas import AuditRecord, ReviewPolicy
from openinterp.core.strict import StrictModel
from openinterp.core.wiring import Wiring

DEFAULT_PATH = Path("docs/analyses/wiring.jsonl")


class WiringRow(StrictModel):
    """One archived experiment's wiring, keyed by its unit and experiment id."""

    slug: str
    experiment_id: str
    weights: Wiring | None = None
    activations: Wiring | None = None


def row_key(record: AuditRecord) -> tuple[str, str]:
    from openinterp.builder.wiki import experiment_id

    return record.submission.identifier.slug, experiment_id(record)


def load_rows(path: Path | None = DEFAULT_PATH) -> dict[tuple[str, str], WiringRow]:
    """The rows in ``path`` by ``(slug, experiment_id)``; empty if it doesn't exist."""
    if path is None or not Path(path).exists():
        return {}
    rows = [
        WiringRow.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return {(r.slug, r.experiment_id): r for r in rows}


def record_wiring(record: AuditRecord, rows: dict) -> tuple[Wiring | None, Wiring | None, bool]:
    """``(weights, activations, backfilled)``: the record's own wiring, else its row's."""
    weights, activations = record.submission.wiring, record.verdict.wiring
    if weights is not None or activations is not None:
        return weights, activations, False
    row = rows.get(row_key(record))
    if row is None:
        return None, None, False
    return row.weights, row.activations, True


def needs_backfill(record: AuditRecord) -> bool:
    return (
        record.verdict is not None
        and record.submission.wiring is None
        and record.verdict.wiring is None
    )


def review_prompts(record: AuditRecord) -> list[str]:
    """The prompts an audit's activation wiring reads: those the unit should fire on."""
    prompts = [p.base_prompt for p in record.submission.counterfactuals]
    prompts += [
        t.prompt for t in record.verdict.adversarial_test_results if t.kind == "positive_holdout"
    ]
    return prompts


def wire(runner, record: AuditRecord) -> WiringRow:
    """Both readings for one archived record."""
    from openinterp.core.exceptions import ArchitectureNotSupportedError
    from openinterp.harness.units import submission_unit
    from openinterp.harness.wiring import activation_wiring, weight_wiring

    sub, verdict = record.submission, record.verdict
    slug, eid = row_key(record)
    unit = submission_unit(sub)
    fraction = (verdict.policy or ReviewPolicy()).activation_threshold
    threshold = fraction * (sub.max_activation or 0.0)
    try:
        weights = weight_wiring(runner, unit)
        activations = (
            activation_wiring(runner, unit, review_prompts(record), threshold)
            if threshold > 0
            else None
        )
    except ArchitectureNotSupportedError:
        weights = activations = None
    return WiringRow(slug=slug, experiment_id=eid, weights=weights, activations=activations)


def backfill(
    records: Iterable[AuditRecord],
    load_runner: Callable[[AuditRecord], object],
    out: Path,
    *,
    limit: int | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """Append a row to ``out`` for each record with no wiring and no row yet, loading
    models in the order records need them. Returns how many rows it wrote."""
    done = load_rows(out)
    todo = [r for r in records if needs_backfill(r) and row_key(r) not in done]
    todo.sort(key=lambda r: _runner_key(r))
    if limit is not None:
        todo = todo[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    runner, loaded = None, None
    for n, record in enumerate(todo, 1):
        if _runner_key(record) != loaded:
            runner, loaded = load_runner(record), _runner_key(record)
        row = wire(runner, record)
        _release_cached_memory()
        with out.open("a") as f:
            f.write(row.model_dump_json() + "\n")
        echo(f"[{n}/{len(todo)}] {row.slug} {row.experiment_id}")
    return len(todo)


def _release_cached_memory() -> None:
    """Hand the accelerator allocator's cached blocks back. Each record's prompts have
    their own shapes, so on MPS the cache otherwise grows record by record until the
    machine swaps (73 GB after ~640 records)."""
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _runner_key(record: AuditRecord) -> tuple:
    """What loading a record's model depends on: name, revision, dtype and whether heads
    need eager attention."""
    from openinterp.harness.units import attn_implementation

    sub = record.submission
    prov = sub.provenance
    return (
        sub.identifier.model_name,
        (prov.model_revision if prov else None) or "",
        (prov.dtype if prov else None) or "float32",
        attn_implementation(sub.identifier) or "",
    )


def sort_rows(path: Path) -> None:
    """Rewrite ``path`` in a stable order, so reruns that append don't churn the diff."""
    rows = sorted(load_rows(path).values(), key=lambda r: (r.slug, r.experiment_id))
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
