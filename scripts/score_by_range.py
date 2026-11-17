"""Where in a pole's activation range the explanation scores look, and the recall there (#3238).

``openinterpretability score`` draws its held-out windows with ``sample_heldout``
(``openinterpretability/harness/explanation_scores.py``). A window is active when it peaks at or
above the pole's 0.25×max line. The active windows are sorted by peak and cut into
equal-count bins, and each bin gives the same number of windows. Each record keeps the bin edges
and the judge's recall in each bin. This script reads them from every record and writes
one row per score, then summary rows:

* ``coverage``: the median bin edges as a share of the pole's maximum activation.
* ``recall``: the mean recall in each bin, pooled and for each judge, with 95% bootstrap
  intervals over records.
* ``top_only``: the detection recall from the top bin alone, against the mean over all bins.
  A score drawn only from the strongest windows would report the first.

It reads ``wiki/`` and writes nothing there::

    uv run python scripts/score_by_range.py --out docs/analyses/score-by-range.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import click

BINS = 5
DRAWS = 2000


def score_rows(wiki_dir: Path) -> list[dict]:
    """One row for each record whose explanation scores keep recall by bin."""
    rows = []
    for path in sorted(wiki_dir.glob("experiments/*/L*/*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        scores = (record.get("verdict") or {}).get("explanation_scores") or {}
        detection = scores.get("detection") or {}
        peak = abs((record.get("submission") or {}).get("max_activation") or 0.0)
        by_bin = detection.get("recall_by_quantile")
        if scores.get("quantiles") != BINS or not by_bin or len(by_bin) != BINS or peak <= 0:
            continue
        edges = scores.get("quantile_edges") or []
        fuzzing = (scores.get("fuzzing") or {}).get("recall_by_quantile")
        rows.append({
            "table": "score", "slug": scores.get("slug"), "record": path.name,
            "judge": scores.get("judge_model"),
            "line": round(scores["threshold"] / peak, 4) if scores.get("threshold") else None,
            "edges": [round(abs(e) / peak, 4) for e in edges] if len(edges) == BINS - 1 else None,
            "detection": by_bin,
            "fuzzing": fuzzing if fuzzing and len(fuzzing) == BINS else None,
        })  # fmt: skip
    return rows


def judge_family(judge: str | None) -> str:
    """A judge's family, without its provider prefix and context suffix."""
    name = (judge or "unknown").removeprefix("external/").split("[")[0]
    for family in ("haiku", "sonnet", "opus", "deepseek", "gemini", "muse", "glm", "gpt"):
        if family in name:
            return family
    return name


def interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    """A 95% bootstrap interval for the mean of ``values``."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(DRAWS))
    return means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS) - 1]


def recall_row(rows: list[dict], kind: str, group: str) -> dict:
    """The mean recall in each bin over ``rows``, with intervals."""
    present = [r[kind] for r in rows if r[kind]]
    out: dict = {"table": "recall", "kind": kind, "group": group, "records": len(present)}
    if present:
        # A bin with no labelled window has no recall (None); it is left out of that bin.
        per_bin = [[r[b] for r in present if r[b] is not None] for b in range(BINS)]
        out["windows_per_bin"] = [len(v) for v in per_bin]
        out["mean"] = [round(statistics.fmean(v), 3) for v in per_bin]
        out["interval"] = [[round(x, 3) for x in interval(v)] for v in per_bin]
    return out


def summary_rows(rows: list[dict]) -> list[dict]:
    """The coverage, recall and top-only rows."""
    out = []
    edges = [r["edges"] for r in rows if r["edges"]]
    lines = [r["line"] for r in rows if r["line"]]
    if edges:
        out.append({
            "table": "coverage", "records": len(edges),
            "line": round(statistics.median(lines), 3) if lines else None,
            "edges": [round(statistics.median(e[b] for e in edges), 3) for b in range(BINS - 1)],
            "top_edge_quartiles": [round(q, 3) for q in
                                   statistics.quantiles([e[-1] for e in edges], n=4)],
        })  # fmt: skip
    for kind in ("detection", "fuzzing"):
        out.append(recall_row(rows, kind, "all"))
        for family in sorted({judge_family(r["judge"]) for r in rows}):
            group = [r for r in rows if judge_family(r["judge"]) == family]
            if len(group) >= 20:
                out.append(recall_row(group, kind, family))
    full = [r["detection"] for r in rows if None not in r["detection"]]
    gaps = [d[-1] - statistics.fmean(d) for d in full]
    out.append({
        "table": "top_only", "records": len(full),
        "top_bin": round(statistics.fmean(d[-1] for d in full), 3),
        "all_bins": round(statistics.fmean(statistics.fmean(d) for d in full), 3),
        "gap": round(statistics.fmean(gaps), 3),
        "gap_interval": [round(x, 3) for x in interval(gaps)],
    })  # fmt: skip
    return out


@click.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("wiki"))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir: Path, out: Path) -> None:
    """Write the score rows and the summary rows."""
    rows = score_rows(wiki_dir)
    if not rows:
        sys.exit("no record keeps recall by bin")
    summary = summary_rows(rows)
    out.write_text("".join(json.dumps(r) + "\n" for r in [*summary, *rows]))
    for row in summary:
        click.echo(json.dumps(row))
    click.echo(f"wrote {len(summary)} summary rows and {len(rows)} score rows → {out}")


if __name__ == "__main__":
    main()
