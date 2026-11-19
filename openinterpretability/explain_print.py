"""What ``explain`` prints for the units of an answer (cli_explain.py).

It prints the table of units: each one's attribution, share and ablation check. Then it
prints the superpositions and circuits, and the strongest paths between the top units.
"""

from __future__ import annotations

import click

from openinterpretability.hubs import hub_line


def print_result(result) -> None:
    """The units, their shares and checks, the groups and the paths, as `explain` prints them."""
    click.echo(f"Answer at position {result.position}: {result.token!r} (logit {result.logit:.2f})")
    click.echo(
        f"{'unit':<14} {'attribution':>11} {'share':>6} {'ablated Δ':>9} {'peak/max':>9}  status"
    )
    for p in result.poles:
        share = abs(p.attribution) / result.total if result.total else 0.0
        ablated = f"{p.ablation_change:+.2f}" if p.ablation_change is not None else ""
        peak = f"{p.strength / p.max_activation:.2f}" if p.max_activation else "—"
        row = f"{p.label:<14} {p.attribution:>+11.3f} {share:>6.1%} {ablated:>9} {peak:>9}"
        click.echo(f"{row}  {p.status}")
        if p.explanation:
            click.echo(f"{'':<14} {p.explanation[:110]}")
        if p.hub:
            click.echo(f"{'':<14} {hub_line(p.hub)}")
    click.echo(
        f"Explained units (VERIFIED, or in a VERIFIED superposition or circuit) hold "
        f"{result.verified_share:.1%} of the total |attribution|; "
        f"{1 - result.verified_share:.1%} is unexplained."
    )
    joint = result.joint
    if joint.get("poles"):
        changed = (
            f"the top token changes to {joint['new_top_token']!r}"
            if joint["top_token_changed"]
            else "the top token stays"
        )
        click.echo(
            f"Mean-ablating the {joint['poles']} VERIFIED units above together moves the answer's "
            f"logit by {joint['logit_change']:+.2f}, and {changed}."
        )
    else:
        click.echo("No unit above is VERIFIED, so nothing here is explained yet.")
    _print_groups(result)
    _print_paths(result)
    click.echo(
        "Attribution is first order and against each unit's mean. The ablated Δ column is "
        "the exact check: mean-ablating a unit moves the logit by about minus its attribution. "
        "Unverified units carry no label."
    )


def _print_groups(result) -> None:
    """The superpositions and circuits, which overlap with the units and aren't in the total."""
    if not result.groups:
        return
    click.echo("Superpositions and circuits (they overlap with the units above):")
    for g in result.groups[:10]:
        ablated = f"{g.ablation_change:+.2f}" if g.ablation_change is not None else ""
        name = f"{g.name} ({g.kind}, {g.members} members)"
        click.echo(f"  {name:<46} {g.attribution:>+9.3f} {ablated:>9}  {g.status}")
        if g.explanation:
            click.echo(f"  {'':<46} {g.explanation[:110]}")


def _print_paths(result, per_target: int = 3) -> None:
    """Each traced unit's strongest incoming edges from the other top units."""
    incoming: dict[str, list[dict]] = {}
    for edge in result.edges:  # sorted by |weight|, strongest first
        incoming.setdefault(edge["target"], []).append(edge)
    if not incoming:
        return
    click.echo("Paths: the top units that drive each unit's contribution (first order).")
    for unit in result.poles:
        edges = incoming.get(unit.label, [])[:per_target]
        if edges:
            sources = ", ".join(f"{e['source']} {e['weight']:+.3f}" for e in edges)
            click.echo(f"  {unit.label:<12} ← {sources}")
