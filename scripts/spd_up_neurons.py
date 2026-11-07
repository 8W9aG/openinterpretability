"""The neuron baseline of GPT-2 small's input projections. It needs no SPD checkpoint.

It reads each layer's ``c_fc`` on the held-out pieces of ``spd_compare.py``. For each ``k``,
it writes the fit after GELU of each token's best ``k`` neurons. Each ``k`` is on a fixed
list, or is an active count in a ``--compare`` file (a ``compare.json``).

    uv run python scripts/spd_up_neurons.py --device cpu --compare run_c.json \\
        --compare run_d.json --out exchange/spd-up-neurons.json
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from spd_compare import STEPS, up_neuron_curve
from spd_model import activation, linear_weight, projection
from train_spd import pieces

MODEL, SEQ_LEN, BATCH_ROWS, ROWS, SEED = "openai-community/gpt2", 128, 8, 64, 0


@click.command()
@click.option("--device", default="cpu", show_default=True)
@click.option("--scan-corpus", type=click.Path(exists=True, path_type=Path), default="corpus.jsonl")
@click.option("--compare", "compares", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(device, scan_corpus, compares, out):
    """Write each input projection's neuron curve to ``--out``."""
    from openinterp.cli import _load_runner

    runner = _load_runner(MODEL, device, "float32", None)
    cfg = SimpleNamespace(scan_corpus=scan_corpus, seq_len=SEQ_LEN, seed=SEED)
    rows, valid, _ = pieces(runner.tokenizer, cfg)
    runs = [json.loads(path.read_text())["layers"] for path in compares]
    hidden = {}
    sites = [(layer, projection(runner, layer, "up")) for layer in range(runner.n_layers)]
    hooks = [module.register_forward_hook(
        lambda _m, args, _o, layer=layer: hidden.__setitem__(layer, args[0].detach().float()))
        for layer, module in sites]  # fmt: skip

    def inputs(chosen: list) -> dict:
        with torch.no_grad():
            runner.model(input_ids=torch.tensor(chosen, device=runner.device))
        return {layer: h[:, 1:].flatten(0, 1) for layer, h in hidden.items()}

    means = {layer: h.mean(0) for layer, h in inputs(rows[-4 * BATCH_ROWS :]).items()}
    held_out = inputs(valid[BATCH_ROWS : BATCH_ROWS + ROWS])
    for hook in hooks:
        hook.remove()
    result = {}
    with torch.no_grad():
        for layer, module in sites:
            w, bias = linear_weight(module)
            w, bias = w.to(runner.device), bias.to(runner.device)
            extra = {round(run[f"{layer}.up"]["spd_l0"]) for run in runs}
            x, offset = held_out[layer] - means[layer], means[layer] @ w + bias
            ks = sorted({*STEPS, *extra})
            curve = up_neuron_curve(w, offset, x, activation(runner, layer), ks)
            result[f"{layer}.up"] = {str(k): round(v, 4) for k, v in curve.items()}
            click.echo(f"L{layer}.up: {json.dumps(result[f'{layer}.up'])}")
    out.write_text(json.dumps({"tokens": held_out[0].shape[0], "curves": result}, indent=1) + "\n")
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
