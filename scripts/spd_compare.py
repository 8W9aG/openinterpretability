"""Compare an SPD checkpoint (``train_spd.py``) with the native neurons it decomposes.

On held-out pieces, per layer, it measures:

* **density at equal fit**: the CI-masked projection's explained variance and active
  subcomponents per token. Then the fit of the same number of neurons per token, and
  the neurons per token that match SPD's fit. Each token keeps its neurons with the
  largest contributions ``|h_i - mean_i|·‖W[i]‖``. That ranking ignores cancellation
  between neurons, so it is not the best choice of neurons.
* **alignment**: each subcomponent's input direction ``V[:, c]`` against the neuron axes.
  Its cosine with the closest neuron is about 0.07 for a random direction in 3,072
  dimensions and 1 for a neuron. ``neuron_spread`` is the effective number of
  subcomponents that share a neuron's row ``W[i]``.
* **drift**: each ``V[:, c]``'s cosine with its value at initialisation. Near 1, the
  subcomponents are still the random directions they started as, and alignment says
  nothing about the model.

    uv run python scripts/spd_compare.py --checkpoint exchange/gpt2-spd/checkpoint.pt \\
        --out exchange/gpt2-spd/compare.json
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import torch
import torch.nn.functional as F
from spd_model import LayerDecomposition, linear_weight
from train_spd import pieces

STEPS = (16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072)


def explained(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """The share of ``y``'s variance over tokens that ``y_hat`` explains."""
    return float(1 - (y - y_hat).pow(2).sum() / (y - y.mean(0)).pow(2).sum())


def top_neurons(x: torch.Tensor, w: torch.Tensor, order: torch.Tensor, k: int) -> float:
    """The explained variance of ``x @ w`` from each token's ``k`` first neurons in ``order``."""
    keep = torch.zeros_like(x).scatter_(1, order[:, :k], 1.0)
    return explained(x @ w, (x * keep) @ w)


def compare_layer(part: LayerDecomposition, v_init: torch.Tensor, h: torch.Tensor) -> dict:
    """Density, alignment and drift for one layer; ``h`` is ``[tokens, d_mlp]``."""
    x = h - part.mean
    y = x @ part.w_target
    inner = x @ part.v
    ci = part.gate_logits(x.unsqueeze(0), inner.unsqueeze(0))[0].clamp(0, 1)
    spd_ev, spd_l0 = explained(y, (inner * ci) @ part.u), float((ci > 0).sum(-1).float().mean())
    order = (x.abs() * part.w_target.norm(dim=1)).argsort(-1, descending=True)
    curve = {k: top_neurons(x, part.w_target, order, k) for k in STEPS}
    best = (part.v / part.v.norm(dim=0)).abs().max(0).values
    share = part.v.abs() * part.u.norm(dim=1)
    spread = share.sum(1).pow(2) / share.pow(2).sum(1)
    drift = F.cosine_similarity(part.v, v_init, dim=0)
    return {
        "spd_explained_variance": round(spd_ev, 3),
        "spd_l0": round(spd_l0, 1),
        "neurons_at_spd_l0": round(top_neurons(x, part.w_target, order, round(spd_l0)), 3),
        "neurons_for_spd_ev": next((k for k in STEPS if curve[k] >= spd_ev), None),
        "neuron_curve": {k: round(v, 3) for k, v in curve.items()},
        "best_neuron_cosine_median": round(float(best.median()), 3),
        "share_best_neuron_cosine_over_0.7": round(float((best > 0.7).float().mean()), 4),
        "neuron_spread_median": round(float(spread.median()), 1),
        "drift_cosine_median": round(float(drift.median()), 3),
    }


def initial_parts(runner, layers: list[int], s: dict) -> dict[int, LayerDecomposition]:
    """The subcomponents as ``train_spd.Decomposer`` initialised them, for the drift."""
    gen, parts = torch.Generator().manual_seed(s["seed"]), {}
    for layer in layers:
        w, bias = linear_weight(runner.down_proj(layer))
        parts[layer] = LayerDecomposition(w.cpu(), None if bias is None else bias.cpu(),
                                          s["components"], s["d_gate"], s["gate"], gen,
                                          s.get("init", "random"))  # fmt: skip
    return parts


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--settings", "settings_json", default=None,
              help="JSON of the run's settings, for a checkpoint saved without them.")  # fmt: skip
@click.option("--rows", type=int, default=64, show_default=True, help="Held-out pieces.")
@click.option("--device", default="auto", show_default=True)
@click.option("--scan-corpus", type=click.Path(exists=True, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(checkpoint, settings_json, rows, device, scan_corpus, out):
    """Write the comparison of each decomposed layer with its neurons to ``--out``."""
    from openinterp.cli import _load_runner

    state = torch.load(checkpoint, map_location="cpu")
    s = state.get("settings") or json.loads(settings_json)
    runner = _load_runner(s["model_name"], device, "float32", s["revision"])
    layers = sorted(state["parts"])
    start = initial_parts(runner, layers, s)
    cfg = SimpleNamespace(scan_corpus=scan_corpus, seq_len=s["seq_len"], seed=s["seed"])
    valid = pieces(runner.tokenizer, cfg)[1]
    ids = torch.tensor(valid[s["batch_rows"] : s["batch_rows"] + rows], device=runner.device)
    hidden = {}
    hooks = [runner.down_proj(layer).register_forward_hook(
        lambda _m, args, _o, layer=layer: hidden.__setitem__(layer, args[0].detach().float()))
        for layer in layers]  # fmt: skip
    with torch.no_grad():
        runner.model(input_ids=ids)
        result = {}
        for layer in layers:
            part = start[layer].to(runner.device)
            v_init = part.v.detach().clone()
            part.load_state_dict(state["parts"][layer])
            result[layer] = compare_layer(part, v_init, hidden[layer][:, 1:].flatten(0, 1))
            click.echo(f"L{layer}: {json.dumps(result[layer])}")
    for hook in hooks:
        hook.remove()
    out.write_text(json.dumps({"step": state["step"], "layers": result}, indent=1) + "\n")
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
