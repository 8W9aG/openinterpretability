"""Attribute one answer of a model to its labelled neuron poles and attention heads.

The command is ``open-interpretability explain``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from openinterp.cli import _load_runner, cli, wiki_dir_option, with_model_options


@cli.command(name="explain")
@click.argument("prompt")
@click.option("--model", "model_name", default="Qwen/Qwen2.5-0.5B", show_default=True)
@click.option("--token", help="The answer token to explain. Default: the model's top token.")
@click.option("--position", default=-1, show_default=True, help="The position to explain.")
@click.option("--top", default=20, show_default=True, help="Units to list.")
@click.option(
    "--edges", default=8, show_default=True, help="Top units whose incoming paths are traced."
)
@click.option(
    "--corpus", default="corpus.jsonl", show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--mean-tokens",
    default=20_000,
    show_default=True,
    help="Tokens of the corpus for each neuron's mean, the attribution's baseline (cached).",
)
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@wiki_dir_option()
@with_model_options
def explain_command(
    prompt, model_name, token, position, top, edges, corpus, mean_tokens, as_json, wiki_dir, device,
    dtype,
):  # fmt: skip
    """Explain the model's answer to PROMPT as fully as the registry allows.

    It attributes the answer's logit to every MLP neuron pole and attention head, against
    each unit's mean on the corpus. It labels the top units from the registry. It reports
    the share of the attribution that VERIFIED units hold, and checks it by mean-ablating
    them. It lists the superpositions and circuits that carry the most attribution. Then
    it traces the paths between the top units.
    """
    from openinterp import explain_groups
    from openinterp.evidence import read_corpus
    from openinterp.explain import explain, registry_labels, unit_means

    runner = _load_runner(model_name, device, dtype)
    means = unit_means(runner, model_name, read_corpus(corpus)[0], mean_tokens)
    labels = registry_labels(Path(wiki_dir), model_name)
    groups = explain_groups.superpositions(model_name) + explain_groups.circuits(model_name)
    group_labels = explain_groups.circuit_labels(Path(wiki_dir), model_name)
    target = None
    if token is not None:
        ids = runner.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise click.BadParameter(
                f"{token!r} is {len(ids)} tokens, not one", param_hint="--token"
            )
        target = ids[0]
    result = explain(
        runner, model_name, prompt, means, labels, token=target, position=position, top=top,
        edges=edges, groups=groups, group_labels=group_labels,
    )  # fmt: skip
    if as_json:
        click.echo(json.dumps(asdict(result) | {"verified_share": result.verified_share}, indent=1))
        return
    _print(result)


def _print(result) -> None:
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
    for g in result.groups:
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
