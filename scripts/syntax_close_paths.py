"""Find what reads the signal of GPT-2's L2 N676+ that round brackets are still open.

The numbers behind *Where the model reads L2 N676+* and *Closing too many* in
papers/syntax-positive-controls/README.md. Each item is a written call with ``depth``
round brackets open, followed by the model's most probable closing token, as in
``syntax_followups.py``. At depths 3 to 5, that token leaves brackets open, and the model
gives a high probability to one more closer. With L2 N676+ ablated, it does not.

- ``close_positions``: ablate the pole at every token, at the closer only, or before it.
- ``close_paths``: patch one later component (an attention head or an MLP) at a time.
  Noising puts the component's output from the ablated run into the clean run. Denoising
  puts the clean output into the ablated run. Both patch every position.
- ``close_done``: read the round-depth poles after the model has closed every bracket,
  and ablate each, to find what keeps the probability of one more closer up.

The measure is the log-odds of a closer: the log of P(a token that starts with ``)``)
over P(any other token).

    P=papers/syntax-positive-controls
    uv run python scripts/syntax_close_paths.py --results $P/followups.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import torch
from syntax_followups import closer_chain, counts_for
from syntax_probes import ROUND_DEPTH_POLES, TARGETS, _rounded, ablate_poles, calls

from openinterp.cli import _load, resolve_device
from openinterp.harness.hooks import pole_strength

MODEL = "openai-community/gpt2"
POLE = ROUND_DEPTH_POLES[0]  # L2 N676+
CHAIN_DEPTHS = (3, 4, 5)  # the depths at which the first closer leaves brackets open
WRITES = ("close_positions", "close_paths", "close_done")
TOP_COMPONENTS = 12


def log_odds(runner, ids: torch.Tensor, **hooks) -> float:
    """Return the log-odds of a closer as the next token after ``ids``."""
    logits = runner.run_with_interventions(ids.unsqueeze(0), **hooks)[0, -1].float()
    probs = logits.softmax(-1).cpu()
    p = float(probs[counts_for(runner, len(probs)) > 0].sum())
    return math.log(max(p, 1e-12) / max(1 - p, 1e-12))


def chain_items(runner, depth: int, closed: bool) -> list[torch.Tensor]:
    """Return each item's token ids after the model's most probable closers.

    With ``closed`` false, the ids end after the first closer, with brackets still open.
    With ``closed``, they end after the chain has closed every bracket. Items whose chain
    does not reach that state are left out.
    """
    out = []
    for k, target in enumerate(TARGETS):
        ids = runner.encode([f"out = {calls(depth, k)}first,{target}"])["input_ids"][0]
        chain = closer_chain(runner, ids, depth)
        steps = chain["steps"]
        want = 1 if not closed else len(steps) - 1
        if want >= len(steps) or (steps[want]["open"] > 0) == closed:
            continue
        added = chain["tokens"][:want]
        out.append(torch.cat([ids, torch.tensor(added, device=ids.device)]))
    return out


def ablate_at(positions: str):
    """Return an intervention that zeroes L2 N676+ at ``positions``: all, last or before."""
    _, neuron, pole = POLE

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        part = act[..., neuron]
        kept = part.clamp(max=0) if pole == "positive" else part.clamp(min=0)
        if positions == "all":
            act[..., neuron] = kept
        elif positions == "last":
            act[..., -1, neuron] = kept[..., -1]
        else:
            act[..., :-1, neuron] = kept[..., :-1]
        return act

    return intervene


def position_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return the mean log-odds of a closer with the pole ablated at each set of positions."""
    rows = []
    layer = POLE[0]
    for depth, found in items.items():
        row = {"kind": "close_positions", "depth": depth, "items": len(found)}
        row["clean"] = sum(log_odds(runner, ids) for ids in found) / len(found)
        for positions in ("all", "last", "before"):
            changes = {layer: ablate_at(positions)}
            row[positions] = sum(log_odds(runner, ids, interventions=changes) for ids in found)
            row[positions] /= len(found)
        rows.append(_rounded(row))
    return rows


def record(cache: dict, key: tuple):
    """Return a hook that stores its input in ``cache[key]`` and changes nothing."""

    def hook(act: torch.Tensor) -> None:
        cache[key] = act.detach().clone()

    return hook


def components(runner) -> list[tuple[str, int, int | None]]:
    """Return every head and MLP after the pole, as (kind, layer, head)."""
    first = POLE[0]
    heads = [
        ("head", layer, h) for layer in range(first, runner.n_layers) for h in range(runner.n_heads)
    ]
    mlps = [("mlp", layer, None) for layer in range(first + 1, runner.n_layers)]
    return [*heads, *mlps]


def run_cached(runner, ids: torch.Tensor, ablated: bool) -> dict:
    """Return every head and MLP output of one run, with the pole ablated or not."""
    cache: dict = {}
    heads = {layer: record(cache, ("head", layer)) for layer in range(runner.n_layers)}
    mlps = {layer: record(cache, ("mlp", layer)) for layer in range(runner.n_layers)}
    hooks = {"head_interventions": heads, "output_interventions": mlps}
    if ablated:
        hooks["interventions"] = {POLE[0]: ablate_at("all")}
    runner.run_with_interventions(ids.unsqueeze(0), **hooks)
    return cache


def patch(source: dict, kind: str, layer: int, head: int | None, width: int) -> dict:
    """Return the hooks that put one component's output from ``source`` into a run."""
    stored = source[(kind, layer)]
    if kind == "mlp":
        return {"output_interventions": {layer: lambda act: stored.to(act.dtype)}}

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        cut = slice(head * width, (head + 1) * width)
        act[..., cut] = stored[..., cut].to(act.dtype)
        return act

    return {"head_interventions": {layer: intervene}}


def path_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return each component's share of the ablation's effect, by noising and denoising.

    ``noise`` is the drop in log-odds when the component alone takes its ablated output,
    over the drop when the pole is ablated. ``denoise`` is the recovery when the
    component alone takes its clean output in the ablated run, over the same drop.
    """
    width = runner.head_dim(0)
    rows = []
    for depth, found in items.items():
        clean_runs = [run_cached(runner, ids, False) for ids in found]
        ablated_runs = [run_cached(runner, ids, True) for ids in found]
        clean = sum(log_odds(runner, ids) for ids in found) / len(found)
        changes = {POLE[0]: ablate_at("all")}
        ablated = sum(log_odds(runner, ids, interventions=changes) for ids in found) / len(found)
        gap = clean - ablated
        shares = []
        for kind, layer, head in components(runner):
            noised = sum(
                log_odds(runner, ids, **patch(a, kind, layer, head, width))
                for ids, a in zip(found, ablated_runs, strict=True)
            ) / len(found)
            restored = []
            for ids, c in zip(found, clean_runs, strict=True):
                hooks = patch(c, kind, layer, head, width)
                hooks["interventions"] = changes
                restored.append(log_odds(runner, ids, **hooks))
            denoised = sum(restored) / len(found)
            name = f"L{layer}H{head}" if kind == "head" else f"L{layer} MLP"
            shares.append(
                {
                    "component": name,
                    "noise": (clean - noised) / gap,
                    "denoise": (denoised - ablated) / gap,
                }
            )
        shares.sort(key=lambda s: -max(s["noise"], s["denoise"]))
        rows.append(
            _rounded(
                {
                    "kind": "close_paths",
                    "depth": depth,
                    "items": len(found),
                    "clean": clean,
                    "ablated": ablated,
                    "top": shares[:TOP_COMPONENTS],
                    "all": shares,
                }
            )
        )
    return rows


def done_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return the poles' strength once every bracket is closed, and each pole's effect.

    ``strength`` is each round-depth pole's mean strength at the last closer. ``ablated``
    is the mean log-odds of one more closer with that pole ablated at every token.
    """
    rows = []
    layers = sorted({layer for layer, _, _ in ROUND_DEPTH_POLES})
    for depth, found in items.items():
        strengths = {f"L{lay} N{n}+": 0.0 for lay, n, _ in ROUND_DEPTH_POLES}
        for ids in found:
            _, cache = runner.run_with_cache(ids.unsqueeze(0), layers)
            for lay, n, pole in ROUND_DEPTH_POLES:
                value = float(pole_strength(cache[lay][0, -1, n], pole))
                strengths[f"L{lay} N{n}+"] += value / len(found)
        row = {
            "kind": "close_done",
            "depth": depth,
            "items": len(found),
            "clean": sum(log_odds(runner, ids) for ids in found) / len(found),
            "strength": strengths,
            "ablated": {},
        }
        for lay, n, pole in ROUND_DEPTH_POLES:
            changes = ablate_poles(((lay, n, pole),))
            mean = sum(log_odds(runner, ids, interventions=changes) for ids in found) / len(found)
            row["ablated"][f"L{lay} N{n}+"] = mean
        rows.append(_rounded(row))
    return rows


@click.command()
@click.option("--results", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
def main(results, device):
    """Replace this script's rows of ``results``, and keep the others."""
    runner = _load(MODEL, resolve_device(device), torch.float32, None, None)
    with torch.no_grad():
        still_open = {d: chain_items(runner, d, closed=False) for d in CHAIN_DEPTHS}
        all_closed = {d: chain_items(runner, d, closed=True) for d in (1, 2, *CHAIN_DEPTHS)}
        rows = [
            *position_rows(runner, still_open),
            *path_rows(runner, still_open),
            *done_rows(runner, all_closed),
        ]
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    kept = [line for line in lines if json.loads(line)["kind"] not in WRITES]
    results.write_text("".join(f"{line}\n" for line in [*kept, *map(json.dumps, rows)]))


if __name__ == "__main__":
    main()
