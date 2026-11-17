"""Do VERIFIED explanations carry to a matched pole of another model? (issue #3189).

A sister study (``audit --sister-of``) audits a pole of one model with the VERIFIED
explanation of a pole of another model, unchanged. This sorts each tested pole by what it
is to the source pole, from ``docs/analyses/sister-neurons.jsonl``:

* ``sister``: the source's sister in that model, a mutual best match above the rotated
  baseline.
* ``match``: another of the source's top matches in that model.
* ``random``: any other pole of that model, which the planner draws as the control.

Each tested pole counts once, by its first study. The rows give the share VERIFIED and
REJECTED, the balanced accuracy, and splits by tokenizer, for each class.

    uv run open-interpretability build --output-dir build/site
    uv run python scripts/explanation_transfer.py --site-dir build/site \\
        --out papers/explanations-across-models/results.jsonl
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import click
import duckdb

from openinterpretability.builder.learned_baseline import fisher_exact, wilson
from openinterpretability.sisters import SisterRow

ROWS = Path("docs/analyses/sister-neurons.jsonl")
KINDS = ("sister", "match", "random")
STUDIES = """
SELECT slug, model_name, sister_of, status, balanced_accuracy, experiment_id
FROM read_parquet('{data}') WHERE sister_of IS NOT NULL ORDER BY reviewed_at, experiment_id
"""


def classify(rows: dict, source: str, model: str, slug: str) -> tuple[str, float | None]:
    """(class, correlation) of the pole ``slug`` of ``model`` against the pole ``source``."""
    row = rows.get((source, model))
    if row is None:
        return "random", None
    if row.sister is not None and row.sister.slug(model) == slug:
        return "sister", row.sister.pearson
    for match in row.matches:
        if match.slug(model) == slug:
            return "match", match.pearson
    return "random", None


def tokenizer_pair(source: str, model: str) -> str:
    """``same`` when both models are Qwen2.5, which share a tokenizer; else ``cross``."""
    return "same" if source.startswith("Qwen--") == model.startswith("Qwen/") else "cross"


def studies(site_dir: Path, rows: dict) -> list[dict]:
    """One row per tested pole, from its first sister study."""
    con = duckdb.connect()
    data = site_dir / "data" / "registry.parquet"
    seen, out = set(), []
    for slug, model, sister_of, status, ba, eid in con.execute(
        STUDIES.format(data=data)
    ).fetchall():
        if slug in seen:
            continue
        seen.add(slug)
        source = sister_of.rsplit("/", 1)[0]
        kind, r = classify(rows, source, model, slug)
        out.append({
            "kind": "study", "class": kind, "source": source, "pole": slug, "experiment": eid,
            "status": status, "balanced_accuracy": ba, "pearson": r,
            "tokenizer": tokenizer_pair(source, model),
        })  # fmt: skip
    return out


def summary(found: list[dict], name: str, tokenizer: str | None = None) -> dict:
    """The share VERIFIED and REJECTED and the balanced accuracy of one class."""
    rows = [s for s in found if s["class"] == name and tokenizer in (None, s["tokenizer"])]
    n = len(rows)
    verified = sum(s["status"] == "VERIFIED" for s in rows)
    rejected = sum(s["status"] == "REJECTED" for s in rows)
    accuracy = [s["balanced_accuracy"] for s in rows if s["balanced_accuracy"] is not None]

    def interval(k):
        return [round(x, 4) for x in wilson(k, n)] if n else None

    return {
        "kind": "class", "class": name, "tokenizer": tokenizer or "all", "studies": n,
        "verified": verified, "verified_ci95": interval(verified),
        "rejected": rejected, "rejected_ci95": interval(rejected),
        "median_balanced_accuracy": round(statistics.median(accuracy), 4) if accuracy else None,
        "mean_balanced_accuracy": round(statistics.mean(accuracy), 4) if accuracy else None,
    }  # fmt: skip


def tests(found: list[dict]) -> dict:
    """Fisher tests of sisters against other matches, and r for VERIFIED and other studies."""
    sisters = [s for s in found if s["class"] == "sister"]
    matches = [s for s in found if s["class"] == "match"]

    def count(rows, status):
        return sum(s["status"] == status for s in rows)

    with_r = [s for s in found if s["pearson"] is not None]
    verified_r = [s["pearson"] for s in with_r if s["status"] == "VERIFIED"]
    other_r = [s["pearson"] for s in with_r if s["status"] != "VERIFIED"]
    return {
        "kind": "tests",
        "verified_p": round(fisher_exact(count(sisters, "VERIFIED"), len(sisters),
                                         count(matches, "VERIFIED"), len(matches)), 4),
        "rejected_p": round(fisher_exact(count(sisters, "REJECTED"), len(sisters),
                                         count(matches, "REJECTED"), len(matches)), 4),
        "median_r_verified": round(statistics.median(verified_r), 4),
        "median_r_other": round(statistics.median(other_r), 4),
        "studies_with_r": len(with_r),
    }  # fmt: skip


@click.command()
@click.option("--site-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(site_dir: Path, out: Path) -> None:
    """Classify every sister study, and write the summary and the rows."""
    rows = {(r.slug, r.other): r for r in map(SisterRow.model_validate_json, ROWS.open())}
    found = studies(site_dir, rows)
    results = [summary(found, k) for k in KINDS]
    results += [summary(found, k, t) for k in ("sister", "match") for t in ("same", "cross")]
    results += [tests(found), *found]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    for r in results[: len(results) - len(found)]:
        click.echo(json.dumps(r))


if __name__ == "__main__":
    main()
