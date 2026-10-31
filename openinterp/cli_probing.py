"""``openinterp probe-compare``: a verified pole against linear probes for its concept.

See :mod:`openinterp.harness.probing` for the probes and :mod:`openinterp.core.probing`
for what is reported.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli, with_model_options


@cli.command("probe-compare")
@click.option("--record", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--train",
    "train_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Labelled JSON lines, {"text": ..., "spans": [[start, end], ...]}: where the '
    "concept holds, marked by a rule that never sees the model.",
)
@click.option(
    "--test",
    "test_path",
    type=click.Path(exists=True, path_type=Path),
    help="The same from another source, to test on. Default: held-out documents of --train.",
)
@click.option("--concept", required=True, help="What the spans mark, as their rule states it.")
@click.option("--test-share", default=0.3, show_default=True, type=click.FloatRange(0, 1))
@click.option("--window-tokens", default=128, show_default=True, type=click.IntRange(2))
@click.option("--max-positive", default=2000, show_default=True, type=click.IntRange(1))
@click.option("--max-negative", default=8000, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Report JSON.")
@with_model_options
def probe_compare(
    record,
    train_path,
    test_path,
    concept,
    test_share,
    window_tokens,
    max_positive,
    max_negative,
    seed,
    out,
    device,
    dtype,
):
    """Compare a neuron pole with logistic, mass-mean and k-sparse probes for its concept,
    on the layer's MLP hidden activations and the residual stream after it.

    Reported, not gated: it shows whether the concept lives in the neuron or is only
    partly carried by it.
    """
    from openinterp.cli import _load_runner
    from openinterp.core.probing import ProbeComparison
    from openinterp.core.schemas import AuditRecord
    from openinterp.harness.probing import compare

    sub = AuditRecord.model_validate_json(record.read_text()).submission
    if sub.identifier.kind != "mlp_neuron_pole" or sub.max_activation is None:
        raise click.ClickException("probe-compare needs a neuron pole with a recorded scale.")
    prov = sub.provenance
    runner = _load_runner(
        sub.identifier.model_name, device, dtype, prov.model_revision if prov else None
    )
    train_wins, test_wins = _split(runner, train_path, test_path, test_share, window_tokens, seed)
    try:
        scores = compare(
            runner,
            sub,
            train_wins,
            test_wins,
            max_positive=max_positive,
            max_negative=max_negative,
            seed=seed,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    report = ProbeComparison(
        concept=concept,
        labels=str(train_path) + (f" → {test_path}" if test_path else ""),
        split="source" if test_path else "document",
        train_source=train_path.name,
        test_source=(test_path or train_path).name,
        window_tokens=window_tokens,
        **scores,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n")
    _echo_summary(report, out)


def _split(runner, train_path: Path, test_path, test_share: float, window_tokens: int, seed):
    """(training windows, test windows): a held-out source, or a split by document."""
    from openinterp.harness.probing import read_labels, split_documents, windows

    train_wins = windows(runner.tokenizer, read_labels(train_path), window_tokens)
    if test_path is None:
        return split_documents(train_wins, test_share, seed)
    return train_wins, windows(runner.tokenizer, read_labels(test_path), window_tokens)


def _echo_summary(report, out: Path) -> None:
    n, dense = report.neuron, report.sites[0].logistic
    click.echo(
        f"{report.slug}: neuron BA {n.balanced_accuracy:.3f}"
        f" (line {n.balanced_accuracy_at_line:.3f}, rank {n.rank})"
        f" vs hidden probe {dense.balanced_accuracy:.3f},"
        f" residual {report.sites[1].logistic.balanced_accuracy:.3f};"
        f" cos {report.sites[0].logistic_cosine:+.2f}; k needed {report.neurons_needed} → {out}"
    )
