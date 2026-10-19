"""Experiment wiki: one Markdown page per completed experiment, plus indexes.

Layout under ``wiki/``::

    README.md                                   home: every experiment, newest first
    experiments/<model>/L<l>/N<n>-<pole>/README.md      neuron page: its experiment history
    experiments/<model>/L<l>/N<n>-<pole>/<id>.md        one experiment (write-once)

Experiment pages are never overwritten: a re-evaluation or revision gets a new
page, so the wiki is the history even though ``audits/`` only holds the latest
record. Each page's "Discussion" section is left for people and agents to edit.
Indexes are regenerated from the pages' front matter on every write.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jinja2 import Environment, PackageLoader, StrictUndefined

from openinterp.core.schemas import ActivatingExample, AuditRecord

ExperimentKind = Literal["review", "reevaluation", "backfill"]

# Advice for each failed gate: what to change before the next submission.
GATE_ADVICE = {
    "enough_hard_negatives": "Re-run the review with more hard negatives (protocol minimum: "
    "{min_hard_negatives}).",
    "enough_holdouts": "Re-run the review with more positive holdouts (protocol minimum: "
    "{min_holdouts}).",
    "specificity": "The hypothesis is too broad: the hard negatives listed below fired. Narrow "
    "the claim to exclude them (skill: `interpretability-illusions-review`).",
    "sensitivity": "The hypothesis does not generalise: the holdouts listed below did not fire. "
    "Narrow the claim to what actually fires, or rescan with longer context and broader data "
    "(skill: `native-neuron-audit`).",
    "counterfactual_reproduction": "Predicted Δ did not reproduce. Re-measure the pairs and "
    "recalibrate the predictions, or replace pairs that test the wrong feature.",
    "lookahead_consistent": "The measured mechanism contradicts the claim. Revisit "
    "retrospective vs lookahead (skill: `vocab-projection-lookahead`).",
    "independent_review": "The proposer and reviewer used the same model. Re-review with a "
    "different reviewer model.",
}


@dataclass(frozen=True)
class WikiEntry:
    path: Path
    meta: dict


def experiment_id(record: AuditRecord) -> str:
    assert record.verdict is not None
    return record.verdict.timestamp.strftime("%Y%m%dT%H%M%SZ")


def neuron_dir(wiki_dir: Path, record: AuditRecord) -> Path:
    return Path(wiki_dir) / "experiments" / record.submission.identifier.slug


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("openinterp.builder", "templates/wiki"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["cell"] = _cell
    env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    env.filters["pct"] = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    return env


def _cell(value: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ⏎ ")


def _context(example: ActivatingExample, before: int = 12, after: int = 4) -> str:
    i = example.max_token_index
    text = (
        "".join(example.tokens[max(0, i - before) : i])
        + f"⟦{example.tokens[i]}⟧"
        + "".join(example.tokens[i + 1 : i + 1 + after])
    )
    return text.replace("```", "ʼʼʼ").replace("\n", "⏎")


def _front_matter(meta: dict) -> str:
    # JSON values are valid YAML, so the block renders on GitHub and parses without PyYAML.
    return "---\n" + "".join(f"{k}: {json.dumps(v)}\n" for k, v in meta.items()) + "---\n"


def _read_front_matter(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line == "---":
            break
        key, _, value = line.partition(": ")
        meta[key] = json.loads(value)
    return meta


def write_experiment_page(
    record: AuditRecord,
    wiki_dir: Path,
    *,
    kind: ExperimentKind = "review",
    record_path: Path | None = None,
    previous_status: str | None = None,
) -> Path | None:
    """Write the page for a completed experiment and refresh the indexes.

    Returns the page path, or ``None`` if the page already exists (pages are
    write-once) or the record has no verdict yet.
    """
    if record.verdict is None:
        return None
    page = neuron_dir(wiki_dir, record) / f"{experiment_id(record)}.md"
    if page.exists():
        return None

    sub, verdict = record.submission, record.verdict
    metrics, policy = verdict.metrics, verdict.policy
    failed = [g for g, ok in (metrics.gates.items() if metrics else []) if not ok]
    advice = [GATE_ADVICE[g].format(**(policy.model_dump() if policy else {})) for g in failed]
    meta = {
        "experiment_id": experiment_id(record),
        "kind": kind,
        "slug": sub.identifier.slug,
        "model": sub.identifier.model_name,
        "layer": sub.identifier.layer,
        "neuron": sub.identifier.neuron_idx,
        "pole": sub.identifier.pole,
        "status": verdict.status,
        "balanced_accuracy": verdict.confidence_score,
        "protocol_version": policy.protocol_version if policy else None,
        "timestamp": verdict.timestamp.isoformat(),
        "hypothesis": sub.natural_language_explanation,
        "proposer": sub.submitting_agent_model,
        "reviewer": verdict.reviewer_agent_model,
    }
    rel_record = (
        Path(os.path.relpath(record_path.resolve(), page.parent.resolve())).as_posix()
        if record_path
        else None
    )
    body = (
        _environment()
        .get_template("experiment.md")
        .render(
            meta=meta,
            sub=sub,
            verdict=verdict,
            metrics=metrics,
            policy=policy,
            failed=failed,
            advice=advice,
            failed_probes=[
                t
                for t in verdict.adversarial_test_results
                if not t.passed and t.kind in {"hard_negative", "positive_holdout"}
            ],
            examples=[(e.max_activation, _context(e)) for e in sub.top_examples[:10]],
            kind=kind,
            previous_status=previous_status,
            record_link=rel_record,
        )
    )
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_front_matter(meta) + "\n" + body, encoding="utf-8")
    rebuild_indexes(wiki_dir)
    return page


def load_entries(wiki_dir: Path) -> list[WikiEntry]:
    root = Path(wiki_dir) / "experiments"
    entries = [
        WikiEntry(path, _read_front_matter(path))
        for path in root.rglob("*.md")
        if path.name != "README.md"
    ]
    return sorted((e for e in entries if e.meta), key=lambda e: e.meta["timestamp"], reverse=True)


def rebuild_indexes(wiki_dir: Path) -> None:
    wiki_dir = Path(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(wiki_dir)
    env = _environment()

    by_neuron: dict[str, list[WikiEntry]] = {}
    for entry in entries:
        by_neuron.setdefault(entry.meta["slug"], []).append(entry)
    for slug, history in by_neuron.items():
        index = wiki_dir / "experiments" / slug / "README.md"
        index.write_text(
            env.get_template("neuron.md").render(
                slug=slug, latest=history[0].meta, history=history
            ),
            encoding="utf-8",
        )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.meta["status"]] = counts.get(entry.meta["status"], 0) + 1
    rows = [
        {
            **e.meta,
            "link": e.path.relative_to(wiki_dir).as_posix(),
            "neuron_link": e.path.parent.relative_to(wiki_dir).as_posix() + "/README.md",
        }
        for e in entries
    ]
    (wiki_dir / "README.md").write_text(
        env.get_template("index.md").render(rows=rows, counts=counts, neurons=len(by_neuron)),
        encoding="utf-8",
    )


def backfill(records: Sequence[tuple[AuditRecord, Path]], wiki_dir: Path) -> list[Path]:
    """Write pages for reviewed records that have none yet (idempotent)."""
    written = [
        write_experiment_page(record, wiki_dir, kind="backfill", record_path=path)
        for record, path in records
    ]
    rebuild_indexes(wiki_dir)
    return [p for p in written if p is not None]
