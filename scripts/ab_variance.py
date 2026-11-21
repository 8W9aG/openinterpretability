"""How to make prompt A/B tests less noisy (issue #3198, docs/analyses/ab-variance.md).

A prompt A/B test compares two arms on the same units. Its noise sets how many units it
needs. This script measures the noise from the reviews of the reviewer crossover and the
verdict-stability study (``exchange/crossover``). Each of their 30 submissions has three
Sonnet reviews at 20 probes and two at 30. Two reviews of one submission on one prompt
differ only by chance, so they form a null A/B test.

    uv run python scripts/ab_variance.py extract   # exchange/crossover -> ab-variance.jsonl
    uv run python scripts/ab_variance.py report    # ab-variance.jsonl -> ab-variance.json
"""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

import click

ROOT = Path(__file__).parents[1]
WORK = ROOT / "exchange" / "crossover"
ROWS = ROOT / "docs" / "analyses" / "ab-variance.jsonl"
REPORT = ROOT / "docs" / "analyses" / "ab-variance.json"
REVIEWS = ("sonnet", "sonnet2", "sonnet3", "sonnet30", "sonnet30b", "opus", "haiku", "original")
TWENTY = ("sonnet", "sonnet2", "sonnet3")  # same reviewer, same prompt, 20 probes
THIRTY = ("sonnet30", "sonnet30b")  # same reviewer, same prompt, 30 probes
MIN_SPECIFICITY, MIN_SENSITIVITY = 0.80, 0.60  # ReviewPolicy's gates
Z_ALPHA, Z_POWER = 1.96, 0.84  # two-sided 5% test at 80% power
SHIFT_BA, SHIFT_VERIFIED = 0.05, 0.10  # the effects an A/B test should detect


def row(submission: str, review: str, result: dict) -> dict:
    """One review's outcome, with its margins to the specificity and sensitivity gates.

    Args:
        submission: The submission's folder name, such as ``c00``.
        review: The review's arm, such as ``sonnet2``.
        result: The review's ``result.json``.

    Returns:
        The row that ``ab-variance.jsonl`` stores.

    >>> r = row("c00", "sonnet", {"balanced_accuracy": 0.9, "outcome": "VERIFIED",
    ...                           "specificity": 0.85, "sensitivity": 0.7, "measured_probes": 20})
    >>> r["verified"], round(r["specificity_margin"], 2), r["probes"]
    (True, 0.05, 20)
    """
    return {
        "submission": submission,
        "review": review,
        "balanced_accuracy": result["balanced_accuracy"],
        "verified": result["outcome"] == "VERIFIED",
        "specificity_margin": result["specificity"] - MIN_SPECIFICITY,
        "sensitivity_margin": result["sensitivity"] - MIN_SENSITIVITY,
        "probes": result.get("measured_probes"),
    }


def units_needed(sd: float, shift: float) -> int:
    """The units a paired test needs to detect ``shift`` at 80% power, two-sided 5%.

    Args:
        sd: The standard deviation of one unit's paired difference.
        shift: The true difference to detect.

    Returns:
        The number of units, rounded up.

    >>> units_needed(0.14, 0.05)
    62
    """
    return int(-(-(((Z_ALPHA + Z_POWER) * sd / shift) ** 2) // 1))


def paired_sd(by_review: dict, a: str, b: str, key: str = "balanced_accuracy") -> float:
    """The SD over submissions of review ``b`` minus review ``a``.

    Args:
        by_review: ``{review: {submission: row}}``.
        a: The first arm.
        b: The second arm.
        key: The outcome to compare.

    Returns:
        The sample standard deviation of the paired differences.
    """
    subs = sorted(set(by_review[a]) & set(by_review[b]))
    return statistics.stdev(float(by_review[b][s][key]) - float(by_review[a][s][key]) for s in subs)


def within_variance(by_review: dict, reviews: tuple[str, ...]) -> float:
    """The pooled variance of balanced accuracy among one submission's replicate reviews.

    Args:
        by_review: ``{review: {submission: row}}``.
        reviews: The replicate reviews, all on one prompt.

    Returns:
        The variance of one review around its submission's mean.
    """
    subs = sorted(set.intersection(*(set(by_review[r]) for r in reviews)))
    values = [[by_review[r][s]["balanced_accuracy"] for r in reviews] for s in subs]
    return statistics.mean(statistics.variance(v) for v in values)


def cuped(y: list[float], x: list[float]) -> dict:
    """CUPED: the variance of ``y`` left after a regression on the covariate ``x``.

    Args:
        y: The outcome, one value per unit.
        x: An earlier measurement of each unit.

    Returns:
        The correlation, the variance before and after, and the share removed.

    >>> out = cuped([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    >>> round(out["removed"], 6)
    1.0
    """
    rho = statistics.correlation(x, y)
    before = statistics.variance(y)
    return {"rho": rho, "before": before, "after": before * (1 - rho**2), "removed": rho**2}


def flip_rate(by_review: dict, reviews: tuple[str, ...]) -> float:
    """The share of replicate pairs whose VERIFIED or not differs.

    Args:
        by_review: ``{review: {submission: row}}``.
        reviews: The replicate reviews.

    Returns:
        The share of (pair, submission) cases that disagree.
    """
    cases = [
        by_review[a][s]["verified"] != by_review[b][s]["verified"]
        for a, b in itertools.combinations(reviews, 2)
        for s in sorted(set(by_review[a]) & set(by_review[b]))
    ]
    return sum(cases) / len(cases)


def near_gate(by_review: dict, reviews: tuple[str, ...], within: float = 0.05) -> float:
    """The share of reviews whose specificity is within ``within`` of its gate.

    Args:
        by_review: ``{review: {submission: row}}``.
        reviews: The reviews to count.
        within: The distance to the gate that counts as near.

    Returns:
        The share of reviews near the specificity gate.
    """
    margins = [by_review[r][s]["specificity_margin"] for r in reviews for s in by_review[r]]
    return sum(abs(m) <= within for m in margins) / len(margins)


def summary(rows: list[dict]) -> dict:
    """Every measure in ``docs/analyses/ab-variance.md``, from the stored rows.

    Args:
        rows: The rows of ``ab-variance.jsonl``.

    Returns:
        The report, as ``ab-variance.json`` stores it.
    """
    by_review: dict = {}
    for r in rows:
        by_review.setdefault(r["review"], {})[r["submission"]] = r
    pairs20 = list(itertools.combinations(TWENTY, 2))
    sd20 = statistics.mean(paired_sd(by_review, a, b) for a, b in pairs20)
    sd30 = paired_sd(by_review, *THIRTY)
    noise = within_variance(by_review, TWENTY)
    verified_sd = statistics.mean(paired_sd(by_review, a, b, "verified") for a, b in pairs20)
    first = [by_review["sonnet"][s]["balanced_accuracy"] for s in sorted(by_review["sonnet"])]
    opus = [by_review["opus"][s]["balanced_accuracy"] for s in sorted(by_review["sonnet"])]
    adjusted = cuped(first, opus)
    unpaired_sd = (2 * statistics.variance(first)) ** 0.5
    replicate_sd = {k: (2 * noise / k) ** 0.5 for k in (1, 2, 3, 5)}  # √(2σ²/k)
    return {
        "submissions": len(by_review["sonnet"]),
        "paired_sd_20": sd20,
        "paired_sd_30": sd30,
        "replicates": {
            str(k): {"sd": sd, "units": units_needed(sd, SHIFT_BA)}
            for k, sd in replicate_sd.items()
        },
        "units_ba_20": units_needed(sd20, SHIFT_BA),
        "units_ba_30": units_needed(sd30, SHIFT_BA),
        "verified_sd": verified_sd,
        "units_verified": units_needed(verified_sd, SHIFT_VERIFIED),
        "flip_rate_20": flip_rate(by_review, TWENTY),
        "near_specificity_gate_20": near_gate(by_review, TWENTY),
        "unpaired_sd": unpaired_sd,
        "units_unpaired": units_needed(unpaired_sd, SHIFT_BA),
        "cuped_opus": adjusted,
        "units_unpaired_cuped": units_needed((2 * adjusted["after"]) ** 0.5, SHIFT_BA),
    }  # fmt: skip


@click.group()
def cli() -> None:
    """Measure the noise of a null prompt A/B test (#3198)."""


@cli.command()
def extract() -> None:
    """Write one row per submission and review from ``exchange/crossover``."""
    rows = []
    for unit in sorted(p for p in WORK.iterdir() if p.is_dir()):
        for review in REVIEWS:
            path = unit / review / "result.json"
            if path.exists():
                rows.append(row(unit.name, review, json.loads(path.read_text())))
    ROWS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"{len(rows)} rows -> {ROWS.relative_to(ROOT)}")


@cli.command()
def report() -> None:
    """Write ``ab-variance.json`` from the stored rows, and print it."""
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    result = summary(rows)
    REPORT.write_text(json.dumps(result, indent=1) + "\n")
    click.echo(json.dumps(result, indent=1))


if __name__ == "__main__":
    cli()
