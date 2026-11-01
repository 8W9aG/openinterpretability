"""Reviewer-effect crossover: how much of a verdict is the reviewer (``TODO.md``).

The registry confounds proposer and reviewer: most records were proposed and reviewed by
one agent pair. Here a fixed set of registry submissions, drawn evenly from three proposer
models, is reviewed again from scratch by several reviewer models on the same prompt, and
every probe set is measured under the current policy
(``docs/analyses/reviewer-crossover.md``).

    uv run python scripts/reviewer_crossover.py prepare
    # each exchange/crossover/cNN/<arm>/review.prompt.md is answered in review.probes.json
    uv run python scripts/reviewer_crossover.py measure --arm sonnet \
        --agent-model external/claude-sonnet-5-5
    uv run python scripts/reviewer_crossover.py measure --arm original
    uv run python scripts/reviewer_crossover.py report --out docs/analyses/reviewer-crossover.jsonl
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import click

WORK = Path("exchange/crossover")
PROPOSERS = (
    "external/deepseek-flash[1m]",
    "external/muse-spark-1.3-contributor",
    "external/claude-opus-5-5",
)
ARMS = ("sonnet", "opus", "haiku")
ORIGINAL = "original"  # the record's own probes, re-measured by the same code
STATUSES = ("VERIFIED", "NEEDS_REVISION", "REJECTED")


@click.group()
def cli() -> None:
    """Reviewer-effect crossover."""


@cli.command()
@click.option("--wiki-dir", default="wiki/experiments", type=click.Path(path_type=Path))
@click.option("--per-proposer", default=10, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--probes", "n_probes", default=20, show_default=True)
def prepare(wiki_dir, per_proposer, seed, n_probes):
    """Sample submissions and write one reviewer prompt per arm."""
    from openinterp.cli_review import write_review_prompt

    picked = sample_records(wiki_dir, per_proposer, seed)
    for i, (path, record) in enumerate(picked):
        out = WORK / f"c{i:02d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "record.txt").write_text(str(path))
        for arm in ARMS:
            if not (out / arm / "review.prompt.md").exists():
                write_review_prompt(
                    record.submission, out / arm / "review.prompt.md", n_probes, False
                )


def sample_records(wiki_dir: Path, per_proposer: int, seed: int):
    """``per_proposer`` random protocol-0.8 native-pole draws per proposer, one per unit."""
    from null_calibration import native_records

    pools: dict[str, dict[str, tuple]] = defaultdict(dict)
    for path, record in native_records(wiki_dir):
        sub = record.submission
        if (
            record.verdict.policy.protocol_version == "0.8"
            and sub.provenance.selection == "random"
            and sub.submitting_agent_model in PROPOSERS
        ):
            pools[sub.submitting_agent_model].setdefault(str(sub.identifier), (path, record))
    rng, seen, picked = random.Random(seed), set(), []
    for proposer in PROPOSERS:
        units = sorted(u for u in pools[proposer] if u not in seen)
        for unit in rng.sample(units, per_proposer):
            seen.add(unit)
            picked.append(pools[proposer][unit])
    return picked


@cli.command()
@click.option("--arm", required=True, type=click.Choice(ARMS + (ORIGINAL,)))
@click.option("--agent-model", help="The arm's reviewer, as it reported itself.")
def measure(arm, agent_model):
    """Measure every answered probe set of one arm on its unit (<arm>/result.json)."""
    from null_calibration import review_probes

    from openinterp.agents.reviewer import ProbeSet, probes_from_verdict
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import AuditRecord

    if arm != ORIGINAL and not agent_model:
        raise click.UsageError("--agent-model is required for a reviewer arm.")
    runners: dict[str, object] = {}
    for unit in sorted(WORK.glob("c*")):
        record = AuditRecord.model_validate_json(
            Path((unit / "record.txt").read_text()).read_text()
        )
        sub, out = record.submission, unit / arm
        if (out / "result.json").exists() or (
            arm != ORIGINAL and not (out / "review.probes.json").exists()
        ):
            continue
        try:
            probes = (
                probes_from_verdict(record.verdict)
                if arm == ORIGINAL
                else ProbeSet.model_validate_json((out / "review.probes.json").read_text()).probes
            )
        except ValueError as exc:  # left for the reviewer to fix; the others go ahead
            click.echo(f"{unit.name}/{arm}: invalid probes, skipped: {exc}", err=True)
            continue
        model = sub.identifier.model_name
        if model not in runners:
            runners[model] = _load_runner(model, "auto", "float32")
        reviewer = record.verdict.reviewer_agent_model if arm == ORIGINAL else agent_model
        result = review_probes(sub, runners[model], probes, reviewer) | {"reviewer": reviewer}
        out.mkdir(exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        click.echo(f"{unit.name}/{arm} {sub.identifier}: {result['outcome']}")


@cli.command()
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def report(out):
    """Write one JSON line per submission and print the analysis's tables."""
    rows = list(load_rows())
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    click.echo(f"Wrote {out} ({len(rows)} submissions)\n")
    click.echo(rate_table(rows))
    click.echo(agreement_table(rows))
    click.echo(variance_table(rows))
    click.echo(gate_table(rows))


def load_rows():
    """Submissions whose every arm is measured, with the record's own verdict."""
    from null_calibration import outcome_row

    from openinterp.core.schemas import AuditRecord

    for unit in sorted(WORK.glob("c*")):
        arms = {a: unit / a / "result.json" for a in ARMS + (ORIGINAL,)}
        if not all(p.exists() for p in arms.values()):
            continue
        path = Path((unit / "record.txt").read_text())
        record = AuditRecord.model_validate_json(path.read_text())
        yield outcome_row(path, record) | {
            "proposer": record.submission.submitting_agent_model,
            "arms": {a: json.loads(p.read_text()) for a, p in arms.items()},
        }


def short(agent: str) -> str:
    return agent.removeprefix("external/").removesuffix("[1m]").removesuffix("-contributor")


def rate_table(rows) -> str:
    """VERIFIED per proposer (rows) and reviewer arm (columns)."""
    cols = (ORIGINAL,) + ARMS
    lines = [
        "| Proposer | n | " + " | ".join(cols) + " |",
        "| --- | --- |" + " --- |" * len(cols),
    ]
    for proposer in PROPOSERS + (None,):
        group = [r for r in rows if proposer in (None, r["proposer"])]
        cells = [f"{sum(r['arms'][c]['outcome'] == 'VERIFIED' for r in group)}" for c in cols]
        name = short(proposer) if proposer else "**All**"
        lines.append(f"| {name} | {len(group)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def agreement_table(rows) -> str:
    """Pairwise agreement between arms: same verdict, same VERIFIED call, Cohen's κ."""
    lines = [
        "| Arms | Same verdict | Same VERIFIED call | κ (VERIFIED) | Mean \\|Δ BA\\| |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a, b in combinations((ORIGINAL,) + ARMS, 2):
        x = [r["arms"][a] for r in rows]
        y = [r["arms"][b] for r in rows]
        same = sum(p["outcome"] == q["outcome"] for p, q in zip(x, y, strict=True))
        vx = [p["outcome"] == "VERIFIED" for p in x]
        vy = [q["outcome"] == "VERIFIED" for q in y]
        dba = [
            abs(p["balanced_accuracy"] - q["balanced_accuracy"]) for p, q in zip(x, y, strict=True)
        ]
        agree = sum(p == q for p, q in zip(vx, vy, strict=True))
        lines.append(
            f"| {a} vs {b} | {same}/{len(rows)} | {agree}/{len(rows)} "
            f"| {kappa(vx, vy):.2f} | {sum(dba) / len(dba):.3f} |"
        )
    return "\n".join(lines) + "\n"


def kappa(x: list[bool], y: list[bool]) -> float:
    n = len(x)
    observed = sum(p == q for p, q in zip(x, y, strict=True)) / n
    px, py = sum(x) / n, sum(y) / n
    expected = px * py + (1 - px) * (1 - py)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def variance_table(rows) -> str:
    """Two-way ANOVA (submission × reviewer, one review per cell) over the new arms."""
    lines = [
        "| Outcome | Submission share | Reviewer share | Residual share | Reviewer F (p) |",
        "| --- | --- | --- | --- | --- |",
    ]
    outcomes = {
        "Balanced accuracy": lambda a: a["balanced_accuracy"],
        "VERIFIED (0/1)": lambda a: float(a["outcome"] == "VERIFIED"),
    }
    for name, value in outcomes.items():
        grid = [[value(r["arms"][a]) for a in ARMS] for r in rows]
        sub, rev, res, f = two_way(grid)
        total = sub + rev + res
        p = permutation_p(grid, lambda g: two_way(g)[3])
        lines.append(
            f"| {name} | {sub / total:.0%} | {rev / total:.0%} | {res / total:.0%} "
            f"| {f:.2f} ({p:.3f}) |"
        )
    grid = [[float(r["arms"][a]["outcome"] == "VERIFIED") for a in ARMS] for r in rows]
    q, p = cochran_q(grid), permutation_p(grid, cochran_q)
    return (
        "\n".join(lines)
        + f"\n\nCochran's Q on VERIFIED across {', '.join(ARMS)}: {q:.2f} (p = {p:.3f})\n"
    )


def two_way(grid: list[list[float]]):
    """Sums of squares for rows, columns and residual, and the column F statistic."""
    n, k = len(grid), len(grid[0])
    mean = sum(map(sum, grid)) / (n * k)
    row_means = [sum(r) / k for r in grid]
    col_means = [sum(r[j] for r in grid) / n for j in range(k)]
    ss_rows = k * sum((m - mean) ** 2 for m in row_means)
    ss_cols = n * sum((m - mean) ** 2 for m in col_means)
    ss_total = sum((v - mean) ** 2 for r in grid for v in r)
    ss_res = ss_total - ss_rows - ss_cols
    df_cols, df_res = k - 1, (n - 1) * (k - 1)
    f = (ss_cols / df_cols) / (ss_res / df_res) if ss_res > 1e-12 else float("inf")
    return ss_rows, ss_cols, ss_res, f


def cochran_q(grid: list[list[float]]) -> float:
    k = len(grid[0])
    cols = [sum(r[j] for r in grid) for j in range(k)]
    rows = [sum(r) for r in grid]
    total = sum(rows)
    denom = k * total - sum(r * r for r in rows)
    return (k - 1) * (k * sum(c * c for c in cols) - total**2) / denom if denom else 0.0


def permutation_p(grid, statistic, n: int = 10_000, seed: int = 0) -> float:
    """P-value of ``statistic`` with the arms shuffled within each submission."""
    rng, observed, hits = random.Random(seed), statistic(grid), 0
    for _ in range(n):
        hits += statistic([rng.sample(row, len(row)) for row in grid]) >= observed - 1e-12
    return (hits + 1) / (n + 1)


def gate_table(rows) -> str:
    """How often each gate fails, per arm."""
    gates = sorted({g for r in rows for a in r["arms"].values() for g in a["gates"]})
    cols = (ORIGINAL,) + ARMS
    lines = ["| Gate failed | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    for gate in gates:
        cells = [str(sum(not r["arms"][c]["gates"].get(gate, True) for r in rows)) for c in cols]
        lines.append(f"| `{gate}` | " + " | ".join(cells) + " |")
    medians = []
    for c in cols:
        ba = sorted(r["arms"][c]["balanced_accuracy"] for r in rows)
        medians.append(f"{ba[len(ba) // 2]:.2f}")
    lines.append("| *median balanced accuracy* | " + " | ".join(medians) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cli()
