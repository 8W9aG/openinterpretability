"""Layer-level CKA between a base model and its instruct model (#5127).

Under instruction tuning, a pole's activation changes most in the middle third of
Qwen2.5-0.5B's layers (`docs/analyses/base-instruct.md`). This script asks whether that is
a change of the whole layer or a change within it. It writes two tables:

* ``layer``: each layer's linear CKA (Kornblith et al., 2019) between the two models'
  MLP activations, over the same sample as ``base_instruct.py``. A null puts the base
  model against the same architecture with random weights. Beside each layer is the
  median correlation of its VERIFIED poles between the two models.
* ``swap``: the instruct model with one third of its blocks replaced by the base
  model's. Each VERIFIED pole is correlated again with the base model. If a third's own
  poles come back, the change is local to those blocks. If later poles come back too,
  the change was carried forward from them.

    uv run python scripts/base_instruct_cka.py --device cuda \\
        --out docs/analyses/base-instruct-cka.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from default_on_poles import random_init_runner  # noqa: E402

from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    layer_columns,
    sample_stats,
    sample_windows,
    verified_records,
)

BASE = "Qwen/Qwen2.5-0.5B"
NULL_SEED = 0


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA of two ``[tokens, features]`` matrices, from 0 to 1.

    Args:
        x: One model's activations.
        y: The other's, on the same tokens.

    Returns:
        ``‖Yᵀ X‖² / (‖Xᵀ X‖ ‖Yᵀ Y‖)`` after centring each feature (Frobenius norms).

    Examples:
        >>> x = torch.randn(50, 4, generator=torch.Generator().manual_seed(0))
        >>> rotation = torch.linalg.qr(torch.randn(4, 4))[0]
        >>> round(linear_cka(x, 3 * x @ rotation), 6)
        1.0
    """
    x = (x - x.mean(0)).double()
    y = (y - y.mean(0)).double()
    cross = (y.T @ x).norm() ** 2
    return round(float(cross / ((x.T @ x).norm() * (y.T @ y).norm())), 6)


def standardised(x: torch.Tensor) -> torch.Tensor:
    """Each feature of ``x`` scaled to unit variance, so that no outlier neuron dominates.

    A feature with no variance is left at 0.

    Args:
        x: ``[tokens, features]``.

    Returns:
        The scaled matrix.

    Examples:
        >>> scaled = standardised(torch.tensor([[1.0, 5.0], [3.0, 5.0]]))
        >>> [round(v, 6) for v in scaled.std(0).tolist()]
        [1.0, 0.0]
    """
    sd = x.std(0)
    return x / sd.where(sd > 0, torch.ones_like(sd))


def layer_rows(base, instruct, null, batches, stats, poles_by_layer) -> list[dict]:
    """One row per layer: CKA with the instruct model and with the null, and the poles' r.

    Args:
        base: The base model.
        instruct: The instruct model.
        null: The base architecture with random weights.
        batches: The sample, from ``sample_windows``.
        stats: Each pole's ``pearson`` between base and instruct, by slug.
        poles_by_layer: The poles' slugs, by layer.

    Returns:
        The rows, in layer order.
    """
    rows = []
    for layer in range(base.n_layers):
        every = list(range(base.mlp_width(layer)))
        a = layer_columns(base, batches, layer, every)
        b = layer_columns(instruct, batches, layer, every)
        rs = [stats[s]["pearson"] for s in poles_by_layer.get(layer, ())]
        rs = [r for r in rs if r is not None]
        rows.append({
            "table": "layer", "layer": layer, "cka": linear_cka(a, b),
            "cka_standardised": linear_cka(standardised(a), standardised(b)),
            "cka_null": linear_cka(a, layer_columns(null, batches, layer, every)),
            "poles": len(rs), "median_pole_pearson": statistics.median(rs) if rs else None,
        })  # fmt: skip
        print(f"layer {layer}: CKA {rows[-1]['cka']}", file=sys.stderr)
    return rows


def thirds(n_layers: int) -> list[list[int]]:
    """The layers in three equal blocks, the last taking any remainder.

    Args:
        n_layers: How many layers the model has.

    Returns:
        Three lists of layer indices.

    Examples:
        >>> thirds(24)[1], thirds(7)
        ([8, 9, 10, 11, 12, 13, 14, 15], [[0, 1], [2, 3], [4, 5, 6]])
    """
    k = n_layers // 3
    return [list(range(0, k)), list(range(k, 2 * k)), list(range(2 * k, n_layers))]


def swap_blocks(target, source, layers: list[int]) -> None:
    """Copy ``source``'s blocks at ``layers`` into ``target``, in place.

    Args:
        target: The runner whose blocks are replaced.
        source: The runner whose blocks are copied.
        layers: The block indices.
    """
    for layer in layers:
        target.layers[layer].load_state_dict(source.layers[layer].state_dict())


def swap_row(block: list[int], stats: dict, records, n_layers: int) -> dict:
    """The median r of the poles in each third, with ``block`` taken from the base model.

    Args:
        block: The layers taken from the base model, or [] for none.
        stats: Each pole's ``pearson`` with the base model, by slug.
        records: The VERIFIED records.
        n_layers: How many layers the model has.

    Returns:
        The row.
    """
    medians = []
    for third in thirds(n_layers):
        rs = [stats[r.submission.identifier.slug]["pearson"] for r in records
              if r.submission.identifier.layer in third]  # fmt: skip
        rs = [r for r in rs if r is not None]
        medians.append(statistics.median(rs) if rs else None)
    poles = {slug: value["pearson"] for slug, value in stats.items()}
    return {"table": "swap", "base_layers": block, "median_pole_pearson_by_third": medians,
            "pole_pearson": poles}  # fmt: skip


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
def main(wiki_dir: Path, out: Path, device: str, base_name: str) -> None:
    """Write the layer and swap tables for ``base_name`` and its instruct model.

    Args:
        wiki_dir: The registry, for the VERIFIED records.
        out: Where the rows go.
        device: Where the models run.
        base_name: The base model; the instruct model is ``<base_name>-Instruct``.
    """
    from openinterpretability.cli import _load_runner

    instruct_name = f"{base_name}-Instruct"
    records = verified_records(wiki_dir, base_name)
    base = _load_runner(base_name, device, "float32", None)
    instruct = _load_runner(instruct_name, device, "float32", None)
    null = random_init_runner(base_name, base.tokenizer, NULL_SEED, device)
    batches = sample_windows(base.tokenizer)
    stats = sample_stats(base, instruct, batches, records)
    by_layer: dict[int, list[str]] = {}
    for r in records:
        by_layer.setdefault(r.submission.identifier.layer, []).append(r.submission.identifier.slug)
    rows = layer_rows(base, instruct, null, batches, stats, by_layer)
    rows.append(swap_row([], stats, records, base.n_layers))  # the instruct model as it is
    for block in thirds(base.n_layers):
        swapped = _load_runner(instruct_name, device, "float32", None)
        swap_blocks(swapped, base, block)
        moved = sample_stats(base, swapped, batches, records)
        first = list(moved)[:3]  # in the log, so a swap that changed nothing shows at once
        print(f"swap {block[0]}-{block[-1]}:", {k: (stats[k]["pearson"], moved[k]["pearson"])
                                                 for k in first}, file=sys.stderr)  # fmt: skip
        rows.append(swap_row(block, moved, records, base.n_layers))
        del swapped
    summary = {"table": "summary", "base": base_name, "instruct": instruct_name,
               "poles": len(records), "sample_tokens": SAMPLE_TOKENS,
               "null_seed": NULL_SEED}  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in [summary, *rows]))
    click.echo(json.dumps(summary))


if __name__ == "__main__":
    main()
