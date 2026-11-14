"""``open-interpretability superpositions``: write superposition definitions (issue #3686).

Each subcommand writes one definition per line. A line is what ``audit --superposition``
reads. :mod:`openinterp.superposition_finders` holds the finders.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterp.cli import _load_runner, cli

out_option = click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="The JSONL file to write: one definition per line.",
)


def _write(out: Path, definitions: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(d) + "\n" for d in definitions), encoding="utf-8")
    click.echo(f"wrote {len(definitions)} definitions → {out}")


@cli.group()
def superpositions() -> None:
    """Find superpositions: sparse directions over one layer's MLP neurons."""


@superpositions.command("from-circuit")
@click.option(
    "--rows",
    "rows_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A circuit's rows, such as docs/analyses/sva-circuit.jsonl.",
)
@click.option("--model", "model", required=True, help="The model that the circuit is on.")
@click.option("--top", default=100, show_default=True, help="The circuit's top neurons by rank.")
@click.option("--min-members", default=2, show_default=True, help="The fewest neurons per layer.")
@out_option
def from_circuit_command(rows_file, model, top, min_members, out):
    """One definition per layer with at least --min-members of the circuit's --top neurons."""
    from openinterp.superposition_finders import definition, from_circuit

    lines = rows_file.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    source = f"{rows_file.as_posix()}, top {top} neurons by attribution (#3686)"
    found = from_circuit(rows, top=top, min_members=min_members, source=source)
    _write(out, [definition(model, layer, profile) for layer, profile in found])


@superpositions.command("grammatical-number")
@click.option("--model", "model", required=True, help="The model to probe.")
@click.option("--layers", required=True, help="Comma-separated layers, e.g. 0,4,19.")
@click.option("--k", default=16, show_default=True, help="Neurons per probe.")
@click.option("--n", default=400, show_default=True, help="Agreement pairs: 2 examples each.")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="auto", show_default=True, help="cpu, cuda, mps or auto.")
@out_option
def grammatical_number_command(model, layers, k, n, seed, device, out):
    """One definition per layer: a sparse probe for the number of the agreeing noun."""
    from openinterp.superposition_finders import definition, grammatical_number

    try:
        layer_list = [int(x) for x in layers.split(",") if x.strip()]
    except ValueError as exc:
        raise click.BadParameter(
            "give layers as integers, e.g. 0,4,19", param_hint="--layers"
        ) from exc
    runner = _load_runner(model, device, "float32", None)
    found = grammatical_number(runner, layer_list, k=k, n=n, seed=seed)
    for layer, _, accuracy in found:
        click.echo(f"layer {layer}: held-out accuracy {accuracy:.3f}")
    _write(out, [definition(model, layer, profile) for layer, profile, _ in found])
