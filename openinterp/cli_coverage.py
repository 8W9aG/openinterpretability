"""The ``coverage`` command: how much of each layer's effect the VERIFIED poles keep."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option, with_model_options


@cli.command()
@click.option("--model", "model_name", required=True, help="Hugging Face model ID.")
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    default="corpus.jsonl",
    show_default=True,
    help="Text for the windows.",
)
@click.option(
    "--windows",
    default=64,
    show_default=True,
    type=click.IntRange(1),
    help="Windows to fit the means on, and as many again to measure the loss on.",
)
@click.option("--window-tokens", default=128, show_default=True, type=click.IntRange(8))
@click.option("--batch", default=4, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--layers", help="Comma-separated layers (default: every layer).")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default="docs/analyses/registry-coverage.jsonl",
    show_default=True,
    help="Rows are appended here, one per layer.",
)
@wiki_dir_option()
@with_model_options
def coverage(model_name, corpus, windows, window_tokens, batch, seed, layers, out, wiki_dir,
             device, dtype):  # fmt: skip
    """Measure the share of each MLP layer's loss effect that the VERIFIED poles keep.

    Reported, not gated. See openinterp/coverage.py for the method.
    """
    from openinterp import __version__
    from openinterp.cli import _load_runner
    from openinterp.cli_evidence import read_corpus
    from openinterp.coverage import layer_row, mean_loss, verified_poles
    from openinterp.coverage import windows as draw_windows

    verified = verified_poles(model_name, wiki_dir)
    runner = _load_runner(model_name, device, dtype)
    texts, corpus_name, corpus_sha, _ = read_corpus(corpus)
    fit, test = draw_windows(runner, texts, windows, window_tokens, seed)
    chosen = [int(x) for x in layers.split(",")] if layers else range(runner.n_layers)
    clean = mean_loss(runner, test, batch)
    click.echo(f"Clean loss {clean:.4f} on {test.numel()} tokens", err=True)
    meta = {"model": model_name, "corpus": corpus_name, "corpus_sha256": corpus_sha,
            "windows": windows, "window_tokens": window_tokens, "seed": seed,
            "dtype": dtype, "openinterp": __version__,
            "date": datetime.now(UTC).strftime("%Y-%m-%d")}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(chosen, 1):
        row = meta | layer_row(
            runner, fit, test, layer, verified.get(layer, set()), clean, batch, seed
        )
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        share = row.get("recovered")
        click.echo(f"[{i}/{len(chosen)}] L{layer}: {row['verified_poles']} VERIFIED poles, "
                   f"recovered {'—' if share is None else f'{share:.3f}'}"
                   f" (random {row.get('recovered_random') or 0:.3f})", err=True)  # fmt: skip
