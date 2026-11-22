"""What ``explain`` prints: the plan, the coverage, the narrative and the JSON (#3743).

Moved out of ``cli_explain`` to keep that module short (#5830).
"""

from __future__ import annotations

import textwrap
from dataclasses import asdict

import click


def _as_json(result, planned, show: int) -> dict:
    """The result, without the full ranking, and the next ``show`` planned experiments."""
    out = asdict(result) | {"verified_share": result.verified_share}
    out.pop("order")
    return out | {
        "target": planned.target, "planned_share": planned.planned_share,
        "blocked": planned.blocked, "reaches": planned.reaches,
        "planned": len(planned.experiments), "milestones": dict(_milestones(planned)),
        "experiments": [asdict(e) for e in planned.experiments[:show]],
        "group_experiments": [asdict(e) for e in planned.groups],
        "blocked_units": planned.blocked_units,
    }  # fmt: skip


def _print_plan(planned, show: int, reach: dict) -> None:
    """The planned experiments: how far they reach, the first ``show``, and what is blocked.

    ``reach`` is the coverage estimate (``explain_coverage.coverage_plan``, #4476).
    """
    from openinterpretability.explain_plan import CAVEAT

    k, target = reach["experiments"], reach["target"]
    goal = (f"about {k} unit experiments reach the {target:.0%} coverage target" if k is not None
            else f"the {target:.0%} coverage target is out of reach: every planned unit "
            f"brings back {reach['reach']:.1%}")  # fmt: skip
    if reach.get("overshoot"):
        goal += "; with every planned unit kept, the answer overshoots the clean model"
    click.echo(f"Plan: {goal} (coverage now {reach['now']:.1%}). There are "
               f"{len(planned.experiments)} unit experiments in all, and "
               f"{len(planned.groups)} for superpositions and circuits.")  # fmt: skip
    if reach.get("chosen"):
        click.echo("  Greedy picks, by joint coverage: " + ", ".join(reach["chosen"][:20]))
    if marks := _milestones(planned):
        click.echo(
            "  Of the |attribution|: "
            + "; ".join(f"{k} experiments reach {m:.0%}" for m, k in marks)
            + "."
        )
    for e in planned.experiments[:show] + planned.groups[:show]:
        held = f"; in {', '.join(e.covered_by)}" if e.covered_by else ""
        gain = f"{e.gain:+.1%} coverage" if e.gain is not None else "gain not measured"
        click.echo(
            f"  {e.target:<20} {gain:<18} {e.share:>6.1%} attr.  {e.status:<14} {e.draw}{held}"
        )
        for line in textwrap.wrap(e.rationale, 96):
            click.echo(f"      {line}")
        for command in e.commands:
            click.echo(f"    $ {command}")
    if (n := len(planned.experiments)) > show:
        click.echo(f"  … and {n - show} more unit experiments (--experiments, or --json).")
    if planned.blocked:
        click.echo(
            f"Blocked: {planned.blocked:.1%} of the attribution, on "
            f"{len(planned.blocked_units)} units that no experiment can raise "
            "(stopped, or no follow-up)."
        )
    click.echo(CAVEAT)


def _milestones(planned, marks=(0.5, 0.8, 0.9, 0.99)) -> list[tuple[float, int]]:
    """How many unit experiments, in rank order, reach each explained share below the target."""
    out, reached, k = [], planned.explained, 0
    for mark in (m for m in marks if planned.explained < m < planned.target):
        while reached < mark and k < len(planned.experiments):
            reached += planned.experiments[k].share
            k += 1
        if reached >= mark:
            out.append((mark, k))
    return out


def reached(share: float, target: float, coverage: dict, ran=None) -> str:
    """The last line: the coverage reached against its target, and the share (#4476).

    Examples:
        >>> reached(0.25, 0.8, {"coverage": 0.5, "random": 0.1})[:46]
        'Reached: prompt coverage 50.0%, target 80%, ag'
        >>> reached(0.0, 1.0, {"coverage": None})
        'Reached: no prompt coverage, target 100%; explained share of |attribution| 0.0%.'
    """
    start = f" (from {ran.share_before:.1%} before the run)" if ran is not None else ""
    found = coverage.get("coverage")
    text = (f"Reached: prompt coverage {found:.1%}{start}, target {target:.0%}" if found is not None
            else f"Reached: no prompt coverage, target {target:.0%}")  # fmt: skip
    if found is not None and coverage.get("random") is not None:
        text += f", against {coverage['random']:.1%} for random units matched by kind and layer"
    return text + f"; explained share of |attribution| {share:.1%}."


def _print_coverage(coverage: dict) -> None:
    """Prompt coverage (explain_coverage.py): what the explained units bring back, causally."""
    share, base = coverage["coverage"], coverage["random"]
    if share is None:
        click.echo("Prompt coverage: ablating every unit doesn't change the answer.")
        return
    against = f"; {coverage['units']} matched random units: {base:.1%}" if base is not None else ""
    click.echo(
        f"Prompt coverage: keeping only the {coverage['units']} explained units, and mean-ablating "
        f"the rest, brings back {share:.1%} of the answer's log-probability{against}."
    )
    if coverage.get("overshoot"):
        click.echo(f"Overshoot: the raw coverage is {coverage['raw']:.1%}. The kept units push "
                   "the answer above the clean model, so the credit stops at 100%.")  # fmt: skip


def _print_narrative(told: dict, level: str) -> None:
    """The long form and the paragraph, or the paragraph alone, with any unsupported citation."""
    if not told:
        return
    narrative = told["narrative"]
    if level == "long":
        click.echo("")
        for section in narrative["sections"]:
            click.echo(f"## {section['title']}\n")
            click.echo(textwrap.fill(section["text"], 100) + "\n")
        click.echo("## In one paragraph\n")
    click.echo(textwrap.fill(narrative["paragraph"], 100))
    click.echo(f"\n(narrated by {told['narrator_model']} from the evidence only)")
    if told["citations_not_in_evidence"]:
        missing = ", ".join(told["citations_not_in_evidence"])
        click.echo(f"Warning: the text cites units that the evidence doesn't hold: {missing}.")
