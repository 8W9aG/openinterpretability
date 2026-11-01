"""Experiment registry and wiki: one record and one Markdown page per experiment.

Layout under ``wiki/``::

    README.md                                   home: every experiment, newest first
    experiments/<model>/L<l>/N<n>-<pole>/README.md      neuron page: its experiment history
    experiments/<model>/L<l>/<dictionary>/F<f>/README.md   dictionary feature page
    experiments/<model>/L<l>/H<h>/...                   the same for an attention head
    experiments/<model>/L<l>/N<n>-<pole>/<id>.md        one experiment (write-once)
    experiments/<model>/L<l>/N<n>-<pole>/<id>.json      its full audit record (write-once)

The experiment, not the neuron, is the unit of storage. The ``<id>.json`` archive is
the registry's only source of truth: a neuron can have any number of experiments,
each adds new files and none edits another's, so parallel experiments on one neuron
never conflict. A neuron's standing (latest or most conclusive verdict) is always
derived from its experiments, never stored. Each page's "Discussion" section is left
for people and agents to edit. Indexes (the ``README.md`` files) are regenerated from
the pages' front matter and are gitignored, never committed: every experiment would
otherwise edit the same index rows, and parallel PRs would conflict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jinja2 import Environment, PackageLoader, StrictUndefined

from openinterp.core.paths import experiments_dir
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
    "independent_review": "The proposer and reviewer used the same model, and this policy "
    "requires them to differ. Re-review with a different reviewer model.",
    "beats_confound_baseline": "A trivial baseline (token identity, position or 'never "
    "active') explains the probes at least as well as the hypothesis, or the scan predates "
    "confound profiling. Rescan with `openinterp audit` and sharpen the claim beyond the "
    "trigger tokens (skill: `interpretability-illusions-review`).",
    "not_default_on": "The unit is active on at least half of its scan's tokens, so no hard "
    "negative of a few words can stay inactive. It can't be audited as a detector; record "
    "it as default-on (docs/PROTOCOL.md §1).",
}


@dataclass(frozen=True)
class WikiEntry:
    path: Path
    meta: dict


def experiment_id(record: AuditRecord) -> str:
    assert record.verdict is not None
    return record.verdict.timestamp.strftime("%Y%m%dT%H%M%SZ")


def unit_label(meta: dict) -> str:
    """``L3/N7+``, ``L5/H1`` or ``L8/F40 (dictionary)``, from a page's front matter."""
    if "head" in meta:
        return f"L{meta['layer']}/H{meta['head']}"
    if "feature" in meta:
        return f"L{meta['layer']}/F{meta['feature']} ({meta['dictionary']})"
    return f"L{meta['layer']}/N{meta['neuron']}{'+' if meta['pole'] == 'positive' else '−'}"


def neuron_dir(wiki_dir: Path, record: AuditRecord) -> Path:
    return experiments_dir(wiki_dir) / record.submission.identifier.slug


def home_link(slug: str) -> str:
    """The wiki home's README from a unit's directory, ``experiments/<slug>``."""
    return "../" * (len(slug.split("/")) + 1) + "README.md"


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
    previous_status: str | None = None,
) -> Path | None:
    """Write the page for a completed experiment and refresh the indexes.

    Returns the page path, or ``None`` if the page already exists (pages are
    write-once) or the record has no verdict yet.
    """
    if record.verdict is None:
        return None
    page = neuron_dir(wiki_dir, record) / f"{experiment_id(record)}.md"
    archive_record(record, wiki_dir)
    if page.exists():
        return None

    meta = _page_meta(record, kind)
    body = _page_body(record, wiki_dir, meta, kind, previous_status)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_front_matter(meta) + "\n" + body, encoding="utf-8")
    rebuild_indexes(wiki_dir)
    return page


def _page_meta(record: AuditRecord, kind: ExperimentKind) -> dict:
    """The page's front matter."""
    sub, verdict = record.submission, record.verdict
    ident, policy = sub.identifier, verdict.policy
    if ident.kind == "attn_head":
        unit = {"head": ident.head_idx}
    elif ident.kind == "dictionary_feature":
        unit = {"dictionary": ident.dictionary, "feature": ident.feature_idx}
    else:
        unit = {"neuron": ident.neuron_idx, "pole": ident.pole}
    meta = {
        "experiment_id": experiment_id(record),
        "kind": kind,
        "slug": ident.slug,
        "model": ident.model_name,
        "layer": ident.layer,
        **unit,
        "status": verdict.status,
        "balanced_accuracy": verdict.confidence_score,
        "protocol_version": policy.protocol_version if policy else None,
        "timestamp": verdict.timestamp.isoformat(),
        "hypothesis": sub.natural_language_explanation,
        "proposer": sub.submitting_agent_model,
        "reviewer": verdict.reviewer_agent_model,
    }
    if sub.contributor:
        meta["proposer_contributor"] = sub.contributor
    if verdict.contributor:
        meta["reviewer_contributor"] = verdict.contributor
    if sub.supersedes:
        meta["supersedes"] = sub.supersedes
    return meta


def _page_body(record: AuditRecord, wiki_dir: Path, meta: dict, kind, previous_status) -> str:
    sub, verdict = record.submission, record.verdict
    metrics, policy = verdict.metrics, verdict.policy
    failed = [g for g, ok in (metrics.gates.items() if metrics else []) if not ok]
    advice = [GATE_ADVICE[g].format(**(policy.model_dump() if policy else {})) for g in failed]
    probe_kinds = {"hard_negative", "positive_holdout"}
    return (
        _environment()
        .get_template("experiment.md")
        .render(
            meta=meta,
            label=unit_label(meta),
            sub=sub,
            verdict=verdict,
            metrics=metrics,
            policy=policy,
            failed=failed,
            advice=advice,
            failed_probes=[
                t
                for t in verdict.adversarial_test_results
                if not t.passed and t.measured and t.kind in probe_kinds
            ],
            examples=[(e.max_activation, _context(e)) for e in sub.top_examples[:10]],
            kind=kind,
            previous_status=previous_status,
            record_link=archive_path(wiki_dir, record).name,
            self_reviewed=record.self_reviewed,
            home=home_link(sub.identifier.slug),
        )
    )


def archive_path(wiki_dir: Path, record: AuditRecord) -> Path:
    return neuron_dir(wiki_dir, record) / f"{experiment_id(record)}.json"


def archive_record(record: AuditRecord, wiki_dir: Path) -> Path:
    """Store the full record of a completed experiment next to its page (write-once)."""
    path = archive_path(wiki_dir, record)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return path


def load_archive_paths(wiki_dir: Path) -> list[tuple[Path, AuditRecord]]:
    """Every archived experiment record with its path, oldest first."""
    root = experiments_dir(wiki_dir)
    pairs = [
        (p, AuditRecord.model_validate_json(p.read_text(encoding="utf-8")))
        for p in sorted(root.rglob("*.json"))
    ]
    return sorted(pairs, key=lambda pr: (pr[1].verdict.timestamp, pr[1].submission.identifier.slug))


def load_archive(wiki_dir: Path) -> list[AuditRecord]:
    """Every archived experiment record, oldest first."""
    return [record for _, record in load_archive_paths(wiki_dir)]


def load_entries(wiki_dir: Path) -> list[WikiEntry]:
    root = experiments_dir(wiki_dir)
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
        index = experiments_dir(wiki_dir) / slug / "README.md"
        latest = history[0].meta
        index.write_text(
            env.get_template("neuron.md").render(
                slug=slug,
                latest=latest,
                label=unit_label(latest),
                history=history,
                home=home_link(slug),
            ),
            encoding="utf-8",
        )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.meta["status"]] = counts.get(entry.meta["status"], 0) + 1
    rows = [
        {
            **e.meta,
            "label": unit_label(e.meta),
            "link": e.path.relative_to(wiki_dir).as_posix(),
            "neuron_link": e.path.parent.relative_to(wiki_dir).as_posix() + "/README.md",
        }
        for e in entries
    ]
    (wiki_dir / "README.md").write_text(
        env.get_template("index.md").render(rows=rows, counts=counts, neurons=len(by_neuron)),
        encoding="utf-8",
    )


def backfill(records: Sequence[AuditRecord], wiki_dir: Path) -> list[Path]:
    """Import reviewed records as experiments (record + page) if not yet present (idempotent)."""
    written = [write_experiment_page(record, wiki_dir, kind="backfill") for record in records]
    rebuild_indexes(wiki_dir)
    return [p for p in written if p is not None]
