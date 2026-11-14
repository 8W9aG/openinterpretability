"""``openinterp explain``: attribute one answer of a model to its labelled neuron poles."""

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
@click.option("--top", default=20, show_default=True, help="Poles to list.")
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
    prompt, model_name, token, position, top, corpus, mean_tokens, as_json, wiki_dir, device, dtype
):
    """Explain the model's answer to PROMPT as fully as the registry allows.

    It attributes the answer's logit to every MLP neuron pole, against each neuron's mean
    on the corpus. It labels the top poles from the registry. It reports the share of
    the attribution that VERIFIED poles hold, and checks it by mean-ablating them.
    """
    from openinterp.evidence import read_corpus
    from openinterp.explain import explain, neuron_means, registry_labels

    runner = _load_runner(model_name, device, dtype)
    means = neuron_means(runner, model_name, read_corpus(corpus)[0], mean_tokens)
    labels = registry_labels(Path(wiki_dir), model_name)
    target = None
    if token is not None:
        ids = runner.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise click.BadParameter(
                f"{token!r} is {len(ids)} tokens, not one", param_hint="--token"
            )
        target = ids[0]
    result = explain(
        runner, model_name, prompt, means, labels, token=target, position=position, top=top
    )
    if as_json:
        click.echo(json.dumps(asdict(result) | {"verified_share": result.verified_share}, indent=1))
        return
    _print(result)


def _print(result) -> None:
    click.echo(f"Answer at position {result.position}: {result.token!r} (logit {result.logit:.2f})")
    click.echo(
        f"{'pole':<14} {'attribution':>11} {'share':>6} {'ablated Δ':>9} {'peak/max':>9}  status"
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
        f"VERIFIED poles hold {result.verified_share:.1%} of the total |attribution|; "
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
            f"Mean-ablating the {joint['poles']} VERIFIED poles above together moves the answer's "
            f"logit by {joint['logit_change']:+.2f}, and {changed}."
        )
    else:
        click.echo("No pole above is VERIFIED, so nothing here is explained yet.")
    click.echo(
        "Attribution is first order and against each neuron's mean. The ablated Δ column is "
        "the exact check: mean-ablating a pole moves the logit by about minus its attribution. "
        "Unverified poles carry no label."
    )
