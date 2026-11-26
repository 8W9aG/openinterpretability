"""A traffic map of pythia-70m: which sites carry the registry's explanations (#5231).

A site is one MLP neuron's write (its column of ``dense_4h_to_h``) or one attention head's
output (its slice of ``attention.dense``). A site's traffic is how far a change there would
move the registry's VERIFIED explanations, to first order:

    traffic(s) = Σ_p use_p · ‖∂R_p/∂θ_s‖

Here R_p is pole p's mean peak strength over its own probes, as a share of its scale.
use_p is the pole's share of active tokens on the corpus sample. The script writes a
``site`` row for every site. The row gives the traffic, the count of poles that the site
reaches and the sum without the use weight. It also gives the diagonal Fisher of the loss
over the site's weights, and the weight norm.

Then it tests the map. It adds Gaussian noise of one fixed norm at each of a sample of
sites. Then it runs the batched keep test of ``scripts/random_delta_keep.py`` on every
VERIFIED pole. Each tested site's row gives the poles that flip and the mean drift of
their probes' peaks.

    uv run python scripts/traffic_map.py --device cuda --out papers/traffic-map/results.jsonl
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

from nf4_keep import run_condition  # noqa: E402
from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    first_position_activations,
    probe_activations,
    probe_texts,
    settle_cached,
)

from openinterpretability.core.schemas import ReviewPolicy  # noqa: E402
from openinterpretability.harness.hooks import pole_strength  # noqa: E402
from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

MODEL = "EleutherAI/pythia-70m"
PER_DECILE = 6  # neuron sites tested from each decile of traffic
TOP = 10  # the neuron sites of most traffic, tested apart from the even draw
SEED = 5231
NOISE_NORM = 1.0  # the Frobenius norm of every test change; see the paper's Method


def site_params(model) -> list[tuple[str, int, int, str, slice]]:
    """Every site: ``(kind, layer, index, parameter name, column slice)``.

    Args:
        model: A GPT-NeoX model.
    """
    out = []
    for layer, block in enumerate(model.gpt_neox.layers):
        mlp = f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight"
        out += [("mlp", layer, j, mlp, slice(j, j + 1))
                for j in range(block.mlp.dense_4h_to_h.in_features)]  # fmt: skip
        attn = f"gpt_neox.layers.{layer}.attention.dense.weight"
        heads = model.config.num_attention_heads
        width = block.attention.dense.in_features // heads
        out += [("head", layer, h, attn, slice(h * width, (h + 1) * width)) for h in range(heads)]
    return out


def column_norms(grad: torch.Tensor, kind: str, heads: int) -> torch.Tensor:
    """The norm of ``grad`` over each site of one weight: a column, or a head's columns.

    Args:
        grad: The gradient of one output weight, ``[d_model, in]``.
        kind: ``mlp`` for a neuron per column, or ``head`` for a head per block of columns.
        heads: The number of heads, for ``head``.
    """
    if kind == "mlp":
        return grad.norm(dim=0)
    return grad.reshape(grad.shape[0], heads, -1).pow(2).sum(dim=(0, 2)).sqrt()


def peak_score(model, ids_list, layer: int, neuron: int, pole: str, skip_first: bool):
    """R_p before scaling: the mean over texts of the pole's peak strength, with a graph.

    Args:
        model: The model.
        ids_list: Each probe text's token ids.
        layer: The pole's layer.
        neuron: The pole's neuron.
        pole: ``positive`` or ``negative``.
        skip_first: Leave each text's first token out.
    """
    got = {}
    mlp = model.gpt_neox.layers[layer].mlp.dense_4h_to_h
    hook = mlp.register_forward_hook(lambda _m, args, _o: got.__setitem__("h", args[0]))
    peaks = []
    try:
        device = mlp.weight.device
        for ids in ids_list:
            model(torch.tensor([ids], device=device))
            strengths = pole_strength(got["h"][0, :, neuron], pole)[int(skip_first) :]
            if strengths.numel():
                peaks.append(strengths.max())
    finally:
        hook.remove()
    return torch.stack(peaks).mean() if peaks else None


def output_weights(model) -> dict[tuple[str, int], torch.Tensor]:
    """Each layer's two output weights, by ``(kind, layer)``.

    Args:
        model: A GPT-NeoX model.
    """
    out = {}
    for layer, block in enumerate(model.gpt_neox.layers):
        out["mlp", layer] = block.mlp.dense_4h_to_h.weight
        out["head", layer] = block.attention.dense.weight
    return out


def traffic(model, tokenizer, records, firsts, use: dict[str, float]) -> dict:
    """Each site's traffic, count and unweighted sum, from one backward pass per pole.

    Args:
        model: The base model.
        tokenizer: Its tokenizer.
        records: The VERIFIED records.
        firsts: The first positions of the base model, for each pole's first-token floor.
        use: Each pole's share of active tokens on the corpus sample, by slug.

    Returns:
        ``{(kind, layer): {"traffic", "count", "unweighted"}}``, each a vector over the
        layer's sites.
    """
    weights = output_weights(model)
    heads = model.config.num_attention_heads
    for p in model.parameters():
        p.requires_grad_(False)
    for w in weights.values():
        w.requires_grad_(True)
    zeros = {k: torch.zeros(heads if k[0] == "head" else w.shape[1]) for k, w in weights.items()}
    out = {k: {"traffic": z.clone(), "count": z.clone(), "unweighted": z.clone()}
           for k, z in zeros.items()}  # fmt: skip
    torch.set_grad_enabled(True)  # the runner may have turned gradients off
    for record in records:
        ident = record.submission.identifier
        policy = record.verdict.policy or ReviewPolicy()
        unit, scale = settle_cached(record, None, policy, firsts)
        ids_list = [tokenizer(t)["input_ids"] for t in probe_texts(record)]
        score = peak_score(model, ids_list, ident.layer, ident.neuron_idx, ident.pole,
                           unit.skip_first)  # fmt: skip
        # A pole at layer 0 reads only the embeddings, so no site's write reaches it.
        if score is None or not score.requires_grad:
            continue
        for w in weights.values():
            w.grad = None
        (score / scale).backward()
        for key, w in weights.items():
            if w.grad is None:  # a layer at or above the pole's: its writes don't reach it
                continue
            norms = column_norms(w.grad, key[0], heads).detach().cpu()
            out[key]["traffic"] += use[ident.slug] * norms
            out[key]["unweighted"] += norms
            out[key]["count"] += (norms > 0).float()
    for w in weights.values():
        w.grad = None
        w.requires_grad_(False)
    return out


def loss_fisher(model, batches: list[torch.Tensor]) -> dict:
    """The diagonal Fisher of the language-model loss, summed over each site's weights.

    It is the sum over batches of the squared gradient of the batch's mean loss, the
    empirical Fisher at the batch level.

    Args:
        model: The base model.
        batches: The corpus sample.

    Returns:
        ``{(kind, layer): vector over the layer's sites}``.
    """
    weights = output_weights(model)
    heads = model.config.num_attention_heads
    total = {k: torch.zeros_like(w) for k, w in weights.items()}
    for w in weights.values():
        w.requires_grad_(True)
    torch.set_grad_enabled(True)  # the runner may have turned gradients off
    for batch in batches:
        ids = batch.to(next(model.parameters()).device)
        for w in weights.values():
            w.grad = None
        model(ids, labels=ids).loss.backward()
        for key, w in weights.items():
            total[key] += w.grad.detach() ** 2
    for w in weights.values():
        w.grad = None
        w.requires_grad_(False)
    return {k: column_norms(t.sqrt(), k[0], heads).pow(2).cpu() for k, t in total.items()}


def pole_use(records, base) -> dict[str, float]:
    """Each pole's share of active tokens on the corpus sample, at its record's line.

    Args:
        records: The VERIFIED records.
        base: The base model's columns on the sample, from ``random_delta_keep.base_columns``.
    """
    out = {}
    for record in records:
        ident = record.submission.identifier
        neurons, columns = base[ident.layer]
        acts = pole_strength(columns[:, neurons.index(ident.neuron_idx)], ident.pole)
        policy = record.verdict.policy or ReviewPolicy()
        line = policy.activation_threshold * record.submission.max_activation
        out[ident.slug] = float((acts >= line).float().mean())
    return out


def sample_sites(site_rows: list[dict], per_decile: int = PER_DECILE, seed: int = SEED,
                 top: int = TOP) -> list[tuple[str, dict]]:  # fmt: skip
    """Neuron sites spread evenly over the deciles of traffic, every head, then the top sites.

    The top sites are the ``top`` neuron sites of most traffic that the decile draw missed.
    They test the map's high end, and the statistics keep them apart from the even draw.

    Args:
        site_rows: The ``site`` rows.
        per_decile: The neuron sites from each decile.
        seed: The seed of the draw.
        top: The number of top neuron sites to add.

    Returns:
        ``(draw, row)`` pairs, where the draw is ``decile``, ``head`` or ``top``.
    """
    gen = torch.Generator().manual_seed(seed)
    neurons = sorted((r for r in site_rows if r["kind"] == "mlp"), key=lambda r: r["traffic"])
    chosen = []
    for d in range(10):
        part = neurons[len(neurons) * d // 10 : len(neurons) * (d + 1) // 10]
        picks = torch.randperm(len(part), generator=gen)[:per_decile].tolist()
        chosen += [("decile", part[i]) for i in sorted(picks)]
    chosen += [("head", r) for r in site_rows if r["kind"] == "head"]
    drawn = {id(r) for _, r in chosen}
    return chosen + [("top", r) for r in reversed(neurons) if id(r) not in drawn][:top]


def site_change(model, row: dict, norm: float, seed: int) -> dict[str, torch.Tensor]:
    """Gaussian noise of Frobenius norm ``norm`` on one site's weights, as a full change.

    Args:
        model: The model.
        row: The site's row.
        norm: The norm of the change.
        seed: The seed of the noise.
    """
    params = dict(model.named_parameters())
    weight = params[row["param"]]
    change = torch.zeros(weight.shape)
    cols = slice(row["start"], row["stop"])
    noise = torch.randn(weight.shape[0], cols.stop - cols.start,
                        generator=torch.Generator().manual_seed(seed))  # fmt: skip
    change[:, cols] = noise / noise.norm() * norm
    return {row["param"]: change}


def site_rows(model, traffic_out: dict, fisher: dict) -> list[dict]:
    """One row for each site: its traffic, count, unweighted sum, loss Fisher and weight norm.

    Args:
        model: The base model.
        traffic_out: From :func:`traffic`.
        fisher: From :func:`loss_fisher`.
    """
    heads = model.config.num_attention_heads
    params = dict(model.named_parameters())
    rows = []
    for kind, layer, index, name, cols in site_params(model):
        key = (kind, layer)
        weight_norm = float(params[name][:, cols].norm())
        rows.append({"table": "site", "kind": kind, "layer": layer, "index": index,
                     "param": name, "start": cols.start, "stop": cols.stop,
                     "traffic": round(float(traffic_out[key]["traffic"][index]), 8),
                     "count": int(traffic_out[key]["count"][index]),
                     "unweighted": round(float(traffic_out[key]["unweighted"][index]), 8),
                     "loss_fisher": float(f"{float(fisher[key][index]):.6g}"),
                     "weight_norm": round(weight_norm, 6), "heads": heads})  # fmt: skip
    return rows


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--norm", default=NOISE_NORM, show_default=True, help="The norm of each change.")
@click.option("--limit", default=0, help="Test only this many of the sampled sites; 0 for all.")
def main(wiki_dir, out, device, norm, limit) -> None:
    """Map each site's traffic, then test it with a change of fixed norm at sampled sites.

    Args:
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each tested site.
        device: The device of the model.
        norm: The Frobenius norm of each test change.
        limit: The number of sampled sites to test, or 0 for all of them.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, MODEL)
    runner = _load_runner(MODEL, device, "float32", None)
    batches = sample_windows(runner.tokenizer)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {r.submission.identifier.slug: cached_readings(runner, r, cache, firsts)
               for r in records}  # fmt: skip
    state = {"on_base": on_base, "cache": cache, "firsts": firsts}
    use = pole_use(records, base)
    sites = site_rows(runner.model, traffic(runner.model, runner.tokenizer, records, firsts, use),
                      loss_fisher(runner.model, batches))  # fmt: skip
    lines = base_rows(records, on_base, MODEL) + sites
    lines += [{"table": "use", "slug": s, "use": round(u, 6)} for s, u in use.items()]
    tested = sample_sites(sites)
    for i, (draw, row) in enumerate(tested[: limit or None]):
        fields = {"model": MODEL, "condition": "site", "draw": draw, "kind": row["kind"],
                  "site_layer": row["layer"], "index": row["index"], "seed": i, "norm": norm,
                  "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
        change = site_change(runner.model, row, norm, SEED + i)
        summary, *poles = run_condition(runner, batches, records, base, state, change, fields)
        drift = [statistics.mean(p["drift"]) for p in poles if p["base_passes"] and p["drift"]]
        summary |= {
            "table": "test",
            "traffic": row["traffic"],
            "mean_drift": round(statistics.mean(drift), 6),
            "flips": summary["reproduced_on_base"] - summary["keeps"],
        }
        lines.append(summary)
        click.echo(
            f"{row['kind']} L{row['layer']} {row['index']}: traffic {row['traffic']:.3g} "
            f"flips {summary['flips']} drift {summary['mean_drift']:.4f}",
            err=True,
        )
        out.write_text("".join(json.dumps(r) + "\n" for r in lines))
    out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
