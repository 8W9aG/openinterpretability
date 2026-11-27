"""Check that the score of scripts/wrapper_prune.py predicts the KL drop of its own neurons (#5163).

For each of the top-scored neurons of the instruct model, this prunes that neuron alone. It
then reads the KL to the base model on the same corpus sample that the score came from. The
score is the first-order drop in that KL, so the measured drop should match it if the first
order holds. The neuron's down column norm is given too, beside the layer's median.

    uv run python scripts/wrapper_prune_check.py --base Qwen/Qwen2.5-0.5B --device cuda \\
        --results papers/wrapper-prune/results.jsonl --out check.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import changed, deltas  # noqa: E402
from wrapper_prune import base_logprobs, down_name, mean_kl, pruned_change  # noqa: E402

from openinterpretability.interference.keep import sample_windows  # noqa: E402


def paper_scores(rows: list[dict]) -> dict[str, list[float]]:
    """The scores of the paper's own run: the write prune on the corpus sample.

    Rows from before the variants of #7071 have no ``unit`` or ``loss``, and count as these.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
    """
    return next(r for r in rows if r["table"] == "scores" and r.get("unit", "write") == "write"
                and r.get("loss", "base") == "base")["scores"]  # fmt: skip


def ranked(scores: dict[str, list[float]], n: int) -> list[tuple[int, int, float]]:
    """The ``n`` neurons with the highest score, as ``(layer, neuron, score)``, highest first.

    Args:
        scores: Each layer's scores, as the ``scores`` row of the results holds them.
        n: How many.
    """
    flat = [(int(layer), i, s) for layer, row in scores.items() for i, s in enumerate(row)]
    return sorted(flat, key=lambda t: (-t[2], t[0], t[1]))[:n]


@click.command()
@click.option("--base", "base_name", required=True, help="The base model, e.g. Qwen/Qwen2.5-0.5B.")
@click.option("--results", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="The rows of scripts/wrapper_prune.py.")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--top", default=10, show_default=True, help="The neurons to check, best first.")
def main(base_name, results, out, device, top) -> None:
    """Prune each top neuron alone, and compare the KL's measured drop with its score.

    Args:
        base_name: The base model. Its instruct model is ``<base_name>-Instruct``.
        results: The rows of ``scripts/wrapper_prune.py``, which hold the scores.
        out: The JSONL file to write, one row for each neuron.
        device: The device of the model.
        top: How many of the top neurons to check.
    """
    from openinterpretability.cli import _load_runner

    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    scores = paper_scores(rows)
    runner = _load_runner(base_name, device, "float32", None)
    other = _load_runner(f"{base_name}-Instruct", "cpu", "float32", None)
    delta = deltas(runner.model, other.model)
    del other
    windows = [row.unsqueeze(0) for b in sample_windows(runner.tokenizer) for row in b]
    base_lp = base_logprobs(runner.model, windows, device)
    with changed(runner.model, delta):
        before = mean_kl(runner.model, windows, base_lp, device)
        norms = {layer: layer_.mlp.down_proj.weight.norm(dim=0).cpu()
                 for layer, layer_ in enumerate(runner.model.model.layers)}  # fmt: skip
    lines = []
    for layer, neuron, score in ranked(scores, top):
        with changed(runner.model, pruned_change(delta, runner.model, [(layer, neuron)])):
            after = mean_kl(runner.model, windows, base_lp, device)
        row = {"table": "single", "model": base_name, "layer": layer, "neuron": neuron,
               "param": down_name(layer), "score": score, "kl_before": round(before, 6),
               "measured_drop": round(before - after, 6),
               "column_norm": round(float(norms[layer][neuron]), 4),
               "layer_median_norm": round(float(norms[layer].median()), 4)}  # fmt: skip
        lines.append(row)
        click.echo(json.dumps(row), err=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
