"""The ``pole-effects`` command: each pole's estimated effect on the loss (issue #3486)."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import click

from openinterp.cli import cli, with_model_options


@cli.command(name="pole-effects")
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
    help="Windows to fit the pole means on, and as many again to measure on.",
)
@click.option("--window-tokens", default=128, show_default=True, type=click.IntRange(8))
@click.option("--batch", default=4, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--top",
    default=2000,
    show_default=True,
    type=click.IntRange(1),
    help="Poles kept in the table, largest estimated effect first.",
)
@click.option(
    "--check",
    default=60,
    show_default=True,
    type=click.IntRange(0),
    help="Poles to measure exactly, half from the top 1% and half uniform.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Default: data/pole-effects/<model>.json, which the planner reads.",
)
@with_model_options
def pole_effects(
    model_name, corpus, windows, window_tokens, batch, seed, top, check, out, device, dtype
):
    """Estimate each MLP pole's effect on the loss, and write the largest for the planner.

    The estimate is attribution patching of single-pole mean ablation
    (openinterp/pole_effects.py). --check measures some poles exactly and reports the
    Spearman correlation with the estimate.
    """
    from openinterp import __version__
    from openinterp.cli import _load_runner
    from openinterp.cli_evidence import read_corpus
    from openinterp.coverage import windows as draw_windows
    from openinterp.planner_effects import table_path
    from openinterp.pole_effects import estimate, top_poles

    runner = _load_runner(model_name, device, dtype)
    texts, corpus_name, corpus_sha, _ = read_corpus(corpus)
    fit, test = draw_windows(runner, texts, windows, window_tokens, seed)
    effects = estimate(runner, fit, test, range(runner.n_layers), batch)
    table = {
        "model": model_name,
        "corpus": corpus_name,
        "corpus_sha256": corpus_sha,
        "windows": windows,
        "window_tokens": window_tokens,
        "seed": seed,
        "dtype": dtype,
        "openinterp": __version__,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "method": "attribution patching of single-pole mean ablation",
        "poles_total": sum(v.numel() for v in effects.values()),
        "check": _check(runner, fit, test, effects, check, seed, batch),
        "poles": top_poles(effects, top),
    }
    out = out or table_path(Path("."), model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + "\n")
    rho = table["check"].get("spearman")
    click.echo(
        f"Wrote {len(table['poles'])} poles to {out}"
        + (f"; Spearman {rho:.2f} on {check} checked poles" if rho is not None else "")
    )


def _check(runner, fit, test, effects, n: int, seed: int, batch: int) -> dict:
    """Measure ``n`` poles exactly, half from the estimate's top 1% and half uniform."""
    import torch

    from openinterp.core.stats import spearman
    from openinterp.coverage import mean_loss
    from openinterp.pole_effects import exact_effect, top_poles

    if n == 0:
        return {}
    rng = random.Random(seed)
    total = sum(v.numel() for v in effects.values())
    head = top_poles(effects, max(n, total // 100))
    sample = [(r["layer"], r["neuron"], r["pole"]) for r in rng.sample(head, n // 2)]
    width = next(iter(effects.values())).numel()
    layers = sorted({layer for layer, _ in effects})
    sample += [
        (rng.choice(layers), rng.randrange(width), rng.choice(("positive", "negative")))
        for _ in range(n - n // 2)
    ]
    clean = mean_loss(runner, test, batch)
    rows = [
        {
            "layer": la,
            "neuron": i,
            "pole": p,
            "estimate": float(effects[(la, p)][i]),
            "exact": exact_effect(runner, fit, test, la, i, p, clean, batch),
        }
        for la, i, p in sample
    ]
    rho = spearman(
        torch.tensor([r["estimate"] for r in rows]), torch.tensor([r["exact"] for r in rows])
    )
    return {"poles": n, "spearman": rho, "clean_loss": clean, "rows": rows}
