"""Verdict stability: how often does the same reviewer, on the same prompt, agree with itself?

Issue #3196, docs/analyses/verdict-stability.md. It reuses the reviewer crossover's 30
submissions (``exchange/crossover``, scripts/reviewer_crossover.py). Each already has one
review by Sonnet, Opus and Haiku. Two more Sonnet reviews, on the identical prompt, give
three reviews by one reviewer. Their spread is pure review noise. The crossover's residual
less that noise is the reviewer × submission interaction.

    uv run python scripts/verdict_stability.py measure --agent-model external/claude-sonnet-5-5
    uv run python scripts/verdict_stability.py report --out docs/analyses/verdict-stability.json

Issue #3509 repeats it with 30 probes a review (20 hard negatives, 10 holdouts):

    uv run python scripts/verdict_stability.py prepare --probes 30 --arms sonnet30,sonnet30b
    uv run python scripts/verdict_stability.py measure --agent-model ... --arms sonnet30,sonnet30b
    uv run python scripts/verdict_stability.py report --arms sonnet30,sonnet30b \
        --out docs/analyses/verdict-stability-30.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from reviewer_crossover import ARMS, WORK, two_way  # noqa: E402

REPEATS = ("sonnet", "sonnet2", "sonnet3")  # the crossover's Sonnet arm, then two repeats
NEW = REPEATS[1:]


@click.group()
def cli() -> None:
    """Verdict stability."""


def arms_option(default: tuple[str, ...]):
    """``--arms a,b``: the repeat arms to work on."""
    return click.option(
        "--arms", default=",".join(default), show_default=True,
        callback=lambda _ctx, _param, value: tuple(value.split(",")),
    )  # fmt: skip


@cli.command()
@click.option("--probes", "n_probes", required=True, type=int)
@arms_option(NEW)
def prepare(n_probes, arms):
    """Write a reviewer prompt that asks for ``--probes`` probes into each new arm."""
    from openinterp.cli_review import write_review_prompt
    from openinterp.core.schemas import AuditRecord

    for unit, arm in itertools.product(sorted(WORK.glob("c*")), arms):
        if not (unit / arm / "review.prompt.md").exists():
            path = Path((unit / "record.txt").read_text())
            record = AuditRecord.model_validate_json(path.read_text())
            (unit / arm).mkdir(exist_ok=True)
            write_review_prompt(record.submission, unit / arm / "review.prompt.md", n_probes, False)


@cli.command()
@click.option("--agent-model", required=True, help="The repeat reviewers, as they reported.")
@arms_option(NEW)
def measure(agent_model, arms):
    """Measure each answered repeat probe set on its unit (<arm>/result.json)."""
    from null_calibration import review_probes

    from openinterp.agents.reviewer import ProbeSet
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import AuditRecord

    runners: dict[str, object] = {}
    for unit, arm in itertools.product(sorted(WORK.glob("c*")), arms):
        out = unit / arm
        if (out / "result.json").exists() or not (out / "review.probes.json").exists():
            continue
        record = AuditRecord.model_validate_json(
            Path((unit / "record.txt").read_text()).read_text()
        )
        try:
            probes = ProbeSet.model_validate_json((out / "review.probes.json").read_text()).probes
        except ValueError as exc:
            click.echo(f"{unit.name}/{arm}: invalid probes, skipped: {exc}", err=True)
            continue
        model = record.submission.identifier.model_name
        if model not in runners:
            runners[model] = _load_runner(model, "auto", "float32")
        result = review_probes(record.submission, runners[model], probes, agent_model)
        (out / "result.json").write_text(json.dumps(result | {"reviewer": agent_model}, indent=2))
        click.echo(f"{unit.name}/{arm}: {result['outcome']}")


def load(arms=(*ARMS, *NEW)) -> list[dict[str, dict]]:
    """For each submission with every arm in ``arms`` measured: {arm: result}."""
    out = []
    for unit in sorted(WORK.glob("c*")):
        paths = {a: unit / a / "result.json" for a in arms}
        if all(p.exists() for p in paths.values()):
            out.append({a: json.loads(p.read_text()) for a, p in paths.items()})
    return out


def pairwise(values: list[list]) -> float:
    """The share of within-row pairs that are equal, over all rows."""
    pairs = [a == b for row in values for a, b in itertools.combinations(row, 2)]
    return sum(pairs) / len(pairs)


def fleiss(values: list[list]) -> float | None:
    """Fleiss' κ for rows of equal length, each value one rater's category."""
    n, k = len(values), len(values[0])
    cats = sorted({v for row in values for v in row})
    p_e = sum((sum(row.count(c) for row in values) / (n * k)) ** 2 for c in cats)
    p_o = sum((sum(row.count(c) ** 2 for c in cats) - k) / (k * (k - 1)) for row in values) / n
    return None if p_e == 1 else (p_o - p_e) / (1 - p_e)


def agreement(rows: list[dict], repeats=REPEATS) -> dict:
    """Same-reviewer agreement on status and on VERIFIED, and on each gate."""
    status = [[r[a]["outcome"] for a in repeats] for r in rows]
    verified = [[s == "VERIFIED" for s in row] for row in status]
    gates = sorted(rows[0][repeats[0]]["gates"])
    return {
        "status_pairwise": pairwise(status), "status_fleiss": fleiss(status),
        "status_all_three": sum(len(set(r)) == 1 for r in status) / len(status),
        "verified_pairwise": pairwise(verified), "verified_fleiss": fleiss(verified),
        "verified_any": sum(any(r) for r in verified),
        "verified_all": sum(all(r) for r in verified),
        "verified_majority": sum(sum(r) >= 2 for r in verified),
        "gate_flip_rate": {g: 1 - pairwise([[r[a]["gates"][g] for a in repeats] for r in rows])
                           for g in gates},
    }  # fmt: skip


def mean_of(rows: list[dict], arms, key: str) -> float:
    """The mean of ``key`` over every arm's result that has it."""
    values = [r[a][key] for r in rows for a in arms if r[a][key] is not None]
    return sum(values) / len(values)


def noise_of(rows: list[dict], repeats, key: str = "balanced_accuracy") -> float:
    """The pooled within-submission variance of ``key`` over the repeat arms."""
    grid = [[r[a][key] for a in repeats] for r in rows]
    grid = [row for row in grid if None not in row]
    return sum(
        sum((v - sum(row) / len(row)) ** 2 for v in row) / (len(row) - 1) for row in grid
    ) / len(grid)


def variance(rows: list[dict]) -> dict:
    """Balanced-accuracy variance: submission, same-reviewer noise and interaction."""
    noise = noise_of(rows, REPEATS)
    grid = [[r[a]["balanced_accuracy"] for a in ARMS] for r in rows]
    ss_rows, ss_cols, ss_res, _ = two_way(grid)
    n, k = len(grid), len(grid[0])
    ms_res = ss_res / ((n - 1) * (k - 1))
    reviewer = max((ss_cols / (k - 1) - ms_res) / n, 0.0)
    submission = max((ss_rows / (n - 1) - ms_res) / k, 0.0)
    interaction = max(ms_res - noise, 0.0)
    total = submission + reviewer + interaction + noise
    return {
        "submissions": n, "same_reviewer_noise": noise, "interaction": interaction,
        "reviewer": reviewer, "submission": submission,
        "shares": {k_: v / total for k_, v in (("submission", submission), ("reviewer", reviewer),
                   ("interaction", interaction), ("same_reviewer_noise", noise))},
    }  # fmt: skip


@cli.command()
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@arms_option(REPEATS)
def report(out, arms):
    """Write agreement, gate flips and the variance split as JSON.

    The variance split needs the crossover's arms, so other ``--arms`` report only the
    same-reviewer noise of balanced accuracy and specificity.
    """
    crossover = arms == REPEATS
    rows = load((*ARMS, *NEW) if crossover else arms)
    if not rows:
        raise click.ClickException("no submission has every arm measured yet")
    result = {"submissions": len(rows), **agreement(rows, arms)}
    for key in ("specificity", "sensitivity"):
        result[f"{key}_mean"] = mean_of(rows, arms, key)
    result["specificity_noise"] = noise_of(rows, arms, "specificity")
    result |= {"variance": variance(rows)} if crossover else {"noise": noise_of(rows, arms)}
    out.write_text(json.dumps(result, indent=1) + "\n")
    click.echo(json.dumps(result, indent=1))


if __name__ == "__main__":
    cli()
