"""``open-interpretability circuits``: circuit ids that survive as the registry grows (#4284)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click

from openinterpretability.cli import cli
from openinterpretability.core.paths import DEFAULT_WIKI_DIR


@cli.group(name="circuits")
def circuits_group() -> None:
    """Circuits: clusters of the wiring graph (docs/circuits-and-hubs.md)."""


@circuits_group.command(name="identify")
@click.option(
    "--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_WIKI_DIR
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/circuits"),
    show_default=True,
)
@click.option("--model", "only", default="", help="One model slug; default: every model.")
def identify_command(wiki_dir: Path, out_dir: Path, only: str) -> None:
    """Match each model's wiring clusters to its stored circuit ids, and record the changes.

    It appends what changed to data/circuits/<model>.jsonl (openinterpretability/circuit_ids.py).
    """
    from openinterpretability import circuit_ids
    from openinterpretability.builder.tree import build_tree
    from openinterpretability.builder.wiki import load_archive
    from openinterpretability.core.errata import load_errata
    from openinterpretability.wiring_backfill import load_rows

    today = datetime.now(UTC).date().isoformat()
    rows = load_rows()
    for model in build_tree(load_archive(wiki_dir), load_errata(wiki_dir)):
        if only and model.slug != only:
            continue
        path = circuit_ids.path_for(model.slug, out_dir)
        old, ever = circuit_ids.load(path)
        result = circuit_ids.update(old, circuit_ids.clusters(model, rows), today, ever)
        circuit_ids.append(path, result.events)
        kinds = ", ".join(sorted({e["event"] for e in result.events})) or "no change"
        click.echo(f"{model.slug}: {len(result.circuits)} circuits ({kinds})")


def _model_runner(model: str, device: str, revision: str | None):
    from openinterpretability.cli import _load_runner

    return _load_runner(model, device, "float32", revision)


@circuits_group.command(name="propose")
@click.option("--model", required=True, help="Hugging Face id, e.g. openai-community/gpt2.")
@click.option("--circuit", "circuit_id", required=True, help="Its id in data/circuits, e.g. C4.")
@click.option("--task", "task_file", required=True, type=click.Path(exists=True, path_type=Path),
              help="A CircuitTask JSON: name, description and prompt pairs.")  # fmt: skip
@click.option("--explanation", required=True, help="What the members compute together.")
@click.option("--agent-model", required=True, help="The model that wrote the hypothesis.")
@click.option(
    "--size", default=100, show_default=True, help="Members after growing by attribution."
)
@click.option("--circuits-dir", type=click.Path(file_okay=False, path_type=Path),
              default=Path("data/circuits"), show_default=True)  # fmt: skip
@click.option("--device", default="auto", show_default=True)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def propose_command(model, circuit_id, task_file, explanation, agent_model, size, circuits_dir,
                    device, out):  # fmt: skip
    """Write a task hypothesis: the circuit's cluster grown by attribution on the task."""
    from openinterpretability import circuit_ids
    from openinterpretability.circuit_review import propose
    from openinterpretability.core.circuit_records import CircuitTask

    circuits, _ = circuit_ids.load(circuit_ids.path_for(model.replace("/", "--"), circuits_dir))
    if circuit_id not in circuits:
        raise click.UsageError(f"{circuit_id} is not a current circuit of {model}")
    task = CircuitTask.model_validate_json(task_file.read_text(encoding="utf-8"))
    base = {"model_name": model, "circuit_id": circuit_id, "explanation": explanation,
            "submitting_agent_model": agent_model}  # fmt: skip
    hypothesis = propose(_model_runner(model, device, None), base, task,
                         sorted(circuits[circuit_id]), size)  # fmt: skip
    out.write_text(hypothesis.model_dump_json(indent=1) + "\n")
    click.echo(f"{len(hypothesis.members)} members → {out}")


@circuits_group.command(name="review")
@click.option("--hypothesis", "hypothesis_file", required=True,
              type=click.Path(exists=True, path_type=Path))  # fmt: skip
@click.option(
    "--probes",
    "probes_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="A CircuitProbes JSON: holdout pairs, negative tasks, the reviewer's model.",
)
@click.option(
    "--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_WIKI_DIR
)
@click.option("--device", default="auto", show_default=True)
def review_command(hypothesis_file, probes_file, wiki_dir, device):
    """Measure a hypothesis on the reviewer's probes, decide it, and archive the record."""
    from openinterpretability.circuit_review import review
    from openinterpretability.core.circuit_records import (
        CircuitHypothesis,
        CircuitPolicy,
        CircuitProbes,
        record_path,
    )

    hypothesis = CircuitHypothesis.model_validate_json(hypothesis_file.read_text(encoding="utf-8"))
    probes = CircuitProbes.model_validate_json(probes_file.read_text(encoding="utf-8"))
    runner = _model_runner(hypothesis.model_name, device, hypothesis.model_revision)
    record = review(runner, hypothesis, probes, CircuitPolicy())
    path = record_path(wiki_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1) + "\n")
    gates = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in record.verdict.gates.items())
    click.echo(f"{record.verdict.status} ({gates}) → {path}")
