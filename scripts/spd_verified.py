"""Compare VERIFIED neurons with other studied neurons in an SPD started from the neurons.

The checkpoint must come from ``train_spd.py --init neurons``, so subcomponent c starts
as neuron c. For each neuron, on held-out pieces, it measures:

* **stay**: the cosine of ``V[:, c]`` with neuron c's axis at the end of training;
* **u_stay**: the cosine of ``U[c]`` with the neuron's row ``W[c]``;
* **ci_mean** and **ci_active**: the mean causal importance, and the share of tokens
  where it is above 0;
* **ci_at_peaks**: for each pole, the mean causal importance on the tokens where the
  neuron's centred activation is largest in that pole's direction.

Each measure becomes a percentile within its layer, so layer effects cancel. The poles
are grouped by their standing in the registry: VERIFIED against every other studied
pole. A permutation test gives the p-value of the difference in mean percentile.

    uv run python scripts/spd_verified.py \\
        --checkpoint exchange/gpt2-spd-neurons-50k/checkpoint.pt \\
        --out exchange/gpt2-spd-neurons-50k/verified.json
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from spd_compare import initial_parts
from spd_model import load_parts
from train_spd import pieces

PEAKS = 32


def percentiles(values: torch.Tensor) -> torch.Tensor:
    """The rank of each value in ``values`` as a share between 0 and 1."""
    return values.argsort().argsort().float() / (len(values) - 1)


def permutation_p(a: torch.Tensor, b: torch.Tensor, draws: int = 10_000) -> float:
    """The two-sided p-value of ``a.mean() - b.mean()`` under random relabelling."""
    both, n = torch.cat([a, b]), len(a)
    gen = torch.Generator().manual_seed(0)
    seen = (a.mean() - b.mean()).abs()
    hits = 0
    for _ in range(draws):
        idx = torch.randperm(len(both), generator=gen)
        hits += int((both[idx[:n]].mean() - both[idx[n:]].mean()).abs() >= seen)
    return (hits + 1) / (draws + 1)


def studied_poles(model_name: str) -> list[tuple[int, int, str, str]]:
    """``(layer, neuron, side, status)`` for every studied neuron pole of the model."""
    from openinterp.builder.compiler import load_records
    from openinterp.builder.standing import native_standings
    from openinterp.builder.tree import build_tree

    models = build_tree(load_records(Path("wiki/experiments")))
    return [(n.layer, n.idx, side, st.status) for n, side, st in native_standings(models)
            if n.model == model_name and n.kind == "mlp_neuron_pole"]  # fmt: skip


def update_stats(st: dict, part, x: torch.Tensor) -> None:
    """Add one batch of centred hidden activations ``x`` (``[tokens, d_mlp]``) to ``st``."""
    ci = part.gate_logits(x.unsqueeze(0), (x @ part.v).unsqueeze(0))[0].clamp(0, 1)
    st["ci_sum"] += ci.sum(0)
    st["ci_active"] += (ci > 0).float().sum(0)
    st["tokens"] += len(x)
    for side, sign in (("positive", 1), ("negative", -1)):
        top, top_ci = st["peaks"][side]
        vals = torch.cat([top, sign * x])
        keep = vals.topk(PEAKS, dim=0).indices
        st["peaks"][side] = (vals.gather(0, keep), torch.cat([top_ci, ci]).gather(0, keep))


def collect_stats(runner, parts: dict, valid: list, chunk: int) -> dict:
    """Causal-importance statistics per layer and neuron over the pieces in ``valid``."""
    dev, d = runner.device, next(iter(parts.values())).v.shape[0]
    stats = {layer: {"ci_sum": torch.zeros(d, device=dev), "ci_active": torch.zeros(d, device=dev),
                     "tokens": 0, "peaks": {side: (torch.full((PEAKS, d), -torch.inf, device=dev),
                                                   torch.zeros(PEAKS, d, device=dev))
                                            for side in ("positive", "negative")}}
             for layer in parts}  # fmt: skip
    hidden = {}
    hooks = [runner.down_proj(layer).register_forward_hook(
        lambda _m, args, _o, layer=layer: hidden.__setitem__(layer, args[0].detach().float()))
        for layer in parts]  # fmt: skip
    with torch.no_grad():
        for start in range(0, len(valid), chunk):
            runner.model(input_ids=torch.tensor(valid[start : start + chunk], device=dev))
            for layer, part in parts.items():
                update_stats(stats[layer], part, hidden[layer][:, 1:].flatten(0, 1) - part.mean)
            click.echo(f"{start + chunk} of {len(valid)} pieces")
    for hook in hooks:
        hook.remove()
    return stats


def layer_measures(part, st: dict) -> dict[str, torch.Tensor]:
    """Each neuron's measures in one layer, on the CPU."""
    v, u = part.v.detach(), part.u.detach()
    idx = torch.arange(len(v), device=v.device)
    measures = {
        "stay": v[idx, idx].abs() / v.norm(dim=0),
        "u_stay": torch.nn.functional.cosine_similarity(u, part.w_target, dim=1),
        "ci_mean": st["ci_sum"] / st["tokens"],
        "ci_active": st["ci_active"] / st["tokens"],
        "ci_at_peaks/positive": st["peaks"]["positive"][1].mean(0),
        "ci_at_peaks/negative": st["peaks"]["negative"][1].mean(0),
    }
    return {k: m.cpu() for k, m in measures.items()}


def compare(raw: dict, poles: list) -> dict:
    """VERIFIED poles against the other studied poles, for each measure."""
    ranks = {layer: {k: percentiles(m) for k, m in ms.items()} for layer, ms in raw.items()}
    verified = sum(p[3] == "VERIFIED" for p in poles)
    result = {"poles": {"VERIFIED": verified, "other": len(poles) - verified}, "measures": {}}
    for name in ("stay", "u_stay", "ci_mean", "ci_active", "ci_at_peaks"):
        groups = {"VERIFIED": ([], []), "other": ([], [])}
        for layer, neuron, side, status in poles:
            key = f"ci_at_peaks/{side}" if name == "ci_at_peaks" else name
            group = groups["VERIFIED" if status == "VERIFIED" else "other"]
            group[0].append(float(ranks[layer][key][neuron]))
            group[1].append(float(raw[layer][key][neuron]))
        ver, oth = (torch.tensor(groups[g][0]) for g in ("VERIFIED", "other"))
        result["measures"][name] = {
            "verified_percentile_mean": round(float(ver.mean()), 3),
            "other_percentile_mean": round(float(oth.mean()), 3),
            "verified_median": round(float(torch.tensor(groups["VERIFIED"][1]).median()), 4),
            "other_median": round(float(torch.tensor(groups["other"][1]).median()), 4),
            "p_value": round(permutation_p(ver, oth), 4),
        }
        click.echo(f"{name}: {json.dumps(result['measures'][name])}")
    return result


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--rows", type=int, default=256, show_default=True, help="Held-out pieces.")
@click.option("--chunk", type=int, default=16, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--scan-corpus", type=click.Path(exists=True, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(checkpoint, rows, chunk, device, scan_corpus, out):
    """Write each measure's VERIFIED-against-other comparison to ``--out``."""
    from openinterp.cli import _load_runner

    state = torch.load(checkpoint, map_location="cpu")
    s = state["settings"]
    if s.get("init") != "neurons":
        raise click.UsageError("The checkpoint must come from --init neurons.")
    runner = _load_runner(s["model_name"], device, "float32", s["revision"])
    saved = {site: p for site, p in load_parts(state["parts"]).items() if site[1] == "down"}
    parts = {}
    for (layer, _), part in initial_parts(runner, list(saved), s).items():
        parts[layer] = part.to(runner.device)
        parts[layer].load_state_dict(saved[(layer, "down")])
    cfg = SimpleNamespace(scan_corpus=scan_corpus, seq_len=s["seq_len"], seed=s["seed"])
    valid = pieces(runner.tokenizer, cfg)[1][s["batch_rows"] : s["batch_rows"] + rows]
    stats = collect_stats(runner, parts, valid, chunk)
    raw = {layer: layer_measures(part, stats[layer]) for layer, part in parts.items()}
    poles = [p for p in studied_poles(s["model_name"]) if p[0] in raw]
    out.write_text(json.dumps(compare(raw, poles), indent=1) + "\n")
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
