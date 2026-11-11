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
- ``close_done``: read the round-depth poles after the model closes every bracket. Then
  ablate each pole, to find what keeps the probability of one more closer up.
- ``close_via_l3h5``: noise L3H5, and hold one later component at its clean output.
- ``close_balance``: scale the push (L2 N676+) or the brake (L2 N1195+).
- ``close_l8_neurons``: rank L8 neurons by attribution, then patch the top ones exactly.

The measure is the log-odds of a closer: the log of P(a token that starts with ``)``)
over P(any other token).

    P=papers/syntax-positive-controls
    uv run python scripts/syntax_close_paths.py --results $P/results.followups.jsonl
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
WRITES = (
    "close_positions",
    "close_paths",
    "close_done",
    "close_via_l3h5",
    "close_balance",
    "close_l8_neurons",
)
TARGET_LAYER = 8  # the MLP that carries most of L3H5's effect at depth 3
EXACT_NEURONS = 10  # neurons checked by exact patching, after attribution
READER = ("head", 3, 5)  # L3H5, the head that carries most of the pole's effect
SCALES = (0.5, 1.5, 2.0)  # factors for the pole strengths in the balance test
BRAKE = ROUND_DEPTH_POLES[1]  # L2 N1195+
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


def mean_log_odds(runner, found, hooks_per_item) -> float:
    """Return the mean log-odds over ``found``, each item run with its own hooks."""
    values = [
        log_odds(runner, ids, **hooks) for ids, hooks in zip(found, hooks_per_item, strict=True)
    ]
    return sum(values) / len(values)


def component_shares(runner, found, clean_runs, ablated_runs, clean, ablated) -> list[dict]:
    """Return each component's noising and denoising share of the ablation's effect."""
    width, gap = runner.head_dim(0), clean - ablated
    changes = {POLE[0]: ablate_at("all")}
    shares = []
    for kind, layer, head in components(runner):
        noised = mean_log_odds(
            runner, found, [patch(a, kind, layer, head, width) for a in ablated_runs]
        )
        restored = [
            patch(c, kind, layer, head, width) | {"interventions": changes} for c in clean_runs
        ]
        denoised = mean_log_odds(runner, found, restored)
        name = f"L{layer}H{head}" if kind == "head" else f"L{layer} MLP"
        shares.append(
            {
                "component": name,
                "noise": (clean - noised) / gap,
                "denoise": (denoised - ablated) / gap,
            }
        )
    shares.sort(key=lambda s: -max(s["noise"], s["denoise"]))
    return shares


def path_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return each component's share of the ablation's effect, by noising and denoising.

    ``noise`` is the drop in log-odds when the component alone takes its ablated output,
    over the drop when the pole is ablated. ``denoise`` is the recovery when the
    component alone takes its clean output in the ablated run, over the same drop.
    """
    rows = []
    for depth, found in items.items():
        clean_runs = [run_cached(runner, ids, False) for ids in found]
        ablated_runs = [run_cached(runner, ids, True) for ids in found]
        clean = mean_log_odds(runner, found, [{}] * len(found))
        changes = {"interventions": {POLE[0]: ablate_at("all")}}
        ablated = mean_log_odds(runner, found, [changes] * len(found))
        shares = component_shares(runner, found, clean_runs, ablated_runs, clean, ablated)
        row = {"kind": "close_paths", "depth": depth, "items": len(found), "clean": clean}
        row |= {"ablated": ablated, "top": shares[:TOP_COMPONENTS], "all": shares}
        rows.append(_rounded(row))
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


def noise_reader(runner, found, ablated_runs, held_per_item=None) -> float:
    """Return the mean log-odds with L3H5 noised, and other components held as given.

    ``held_per_item`` gives, for each item, extra hooks (by hook kind, then layer) that
    hold a later component at its clean value. They join the hooks that noise L3H5.
    """
    width = runner.head_dim(0)
    _, layer, head = READER
    hooks_per_item = []
    for k, a in enumerate(ablated_runs):
        hooks = patch(a, "head", layer, head, width)
        for name, by_layer in (held_per_item[k] if held_per_item else {}).items():
            hooks[name] = {**hooks.get(name, {}), **by_layer}
        hooks_per_item.append(hooks)
    return mean_log_odds(runner, found, hooks_per_item)


def via_reader_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return the share of L3H5's effect that flows through each later component.

    Noising L3H5 alone (its output from the ablated run) lowers the log-odds by a drop.
    Holding one later component at its clean output at the same time leaves a smaller
    drop. The share is the part of the drop that the held component removes.
    """
    width, layer = runner.head_dim(0), READER[1]
    rows = []
    for depth, found in items.items():
        clean_runs = [run_cached(runner, ids, False) for ids in found]
        ablated_runs = [run_cached(runner, ids, True) for ids in found]
        clean = mean_log_odds(runner, found, [{}] * len(found))
        drop = clean - noise_reader(runner, found, ablated_runs)
        shares = []
        for kind, at, h in components(runner):
            if at < layer or (kind == "head" and at == layer):
                continue  # only components that run after the reader can carry it
            held = [patch(c, kind, at, h, width) for c in clean_runs]
            after = clean - noise_reader(runner, found, ablated_runs, held)
            name = f"L{at}H{h}" if kind == "head" else f"L{at} MLP"
            shares.append({"component": name, "share": (drop - after) / drop})
        shares.sort(key=lambda x: -x["share"])
        row = {"kind": "close_via_l3h5", "depth": depth, "items": len(found), "drop": drop}
        rows.append(_rounded(row | {"top": shares[:TOP_COMPONENTS]}))
    return rows


def scale_pole(pole, factor: float):
    """Return an intervention that multiplies one pole's strength by ``factor``."""
    _, neuron, sign = pole

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        part = act[..., neuron]
        kept = part.clamp(max=0) if sign == "positive" else part.clamp(min=0)
        act[..., neuron] = kept + (part - kept) * factor
        return act

    return intervene


def balance_rows(runner, open_items: dict, closed_items: dict) -> list[dict]:
    """Return the log-odds of a closer with the push or the brake scaled.

    The push is L2 N676+ and the brake is L2 N1195+. ``open`` is after the first closer,
    with brackets still open: the model should close more. ``closed`` is after every
    bracket is closed: the model should not.
    """
    rows = []
    for depth in CHAIN_DEPTHS:
        row = {"kind": "close_balance", "depth": depth}
        for state, items in (("open", open_items[depth]), ("closed", closed_items[depth])):
            row[state] = {"model": sum(log_odds(runner, ids) for ids in items) / len(items)}
            for name, pole in (("push", POLE), ("brake", BRAKE)):
                for factor in SCALES:
                    changes = {pole[0]: scale_pole(pole, factor)}
                    value = sum(log_odds(runner, ids, interventions=changes) for ids in items)
                    row[state][f"{name} x{factor}"] = value / len(items)
        rows.append(_rounded(row))
    return rows


def attribution_of(runner, found, ablated_runs) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return each L8 neuron's attribution, and each item's clean L8 activations.

    Take the run with L3H5 noised. For each neuron, multiply the gradient of the log-odds
    by the change of its activation from that run to the clean run. Sum over positions.
    """
    width = runner.head_dim(0)
    _, layer, head = READER
    attribution = torch.zeros(runner.mlp_width(TARGET_LAYER))
    clean_acts = []
    for ids, a in zip(found, ablated_runs, strict=True):
        _, cache = runner.run_with_cache(ids.unsqueeze(0), [TARGET_LAYER])
        clean_acts.append(cache[TARGET_LAYER][0].float().cpu())
        store: dict = {}

        def keep(act: torch.Tensor, store=store) -> torch.Tensor:
            store["act"] = act.detach().clone().requires_grad_(True)
            return store["act"]

        hooks = patch(a, "head", layer, head, width)
        with torch.enable_grad(), runner.hooks(interventions={TARGET_LAYER: keep}, **hooks):
            probs = runner.forward(runner.encode(ids.unsqueeze(0)))[0, -1].float().softmax(-1)
            mask = (counts_for(runner, len(probs)) > 0).to(probs.device)
            p = probs[mask].sum().clamp(1e-12, 1 - 1e-12)
            torch.log(p / (1 - p)).backward()
        act = store["act"]
        change = clean_acts[-1] - act.detach()[0].float().cpu()
        attribution += (act.grad[0].float().cpu() * change).sum(0) / len(found)
    return attribution, clean_acts


def hold_neuron(clean_act: torch.Tensor, neuron: int) -> dict:
    """Return the hooks that hold one L8 neuron at its clean activation."""

    def restore(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        act[0, :, neuron] = clean_act[:, neuron].to(act.dtype).to(act.device)
        return act

    return {"interventions": {TARGET_LAYER: restore}}


def neuron_rows(runner, items: dict[int, list[torch.Tensor]]) -> list[dict]:
    """Return the L8 neurons that carry L3H5's effect, by attribution, then exact patching.

    Take the top ``EXACT_NEURONS`` neurons by attribution (see :func:`attribution_of`),
    and L8 N689. Hold each at its clean activation, one at a time, while L3H5 is noised.
    ``share`` is the part of the drop that holding the neuron removes.
    """
    poles_of_interest = {n for lay, n, _ in ROUND_DEPTH_POLES if lay == TARGET_LAYER}
    rows = []
    for depth, found in items.items():
        ablated_runs = [run_cached(runner, ids, True) for ids in found]
        attribution, clean_acts = attribution_of(runner, found, ablated_runs)
        clean = mean_log_odds(runner, found, [{}] * len(found))
        drop = clean - noise_reader(runner, found, ablated_runs)
        top = [int(i) for i in attribution.abs().topk(EXACT_NEURONS).indices]
        checked = []
        for n in dict.fromkeys([*top, *poles_of_interest]):
            held = [hold_neuron(c, n) for c in clean_acts]
            share = (drop - (clean - noise_reader(runner, found, ablated_runs, held))) / drop
            checked.append(
                {"neuron": n, "attribution": float(attribution[n]) / drop, "share": share}
            )
        checked.sort(key=lambda x: -x["share"])
        ranks = {f"N{n}": int((attribution > attribution[n]).sum()) for n in poles_of_interest}
        row = {"kind": "close_l8_neurons", "depth": depth, "items": len(found), "drop": drop}
        row |= {"attribution_total": float(attribution.sum()) / drop, "checked": checked}
        rows.append(_rounded(row | {"pole_ranks": ranks}))
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
            *via_reader_rows(runner, still_open),
            *balance_rows(runner, still_open, all_closed),
            *neuron_rows(runner, still_open),
        ]
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    kept = [line for line in lines if json.loads(line)["kind"] not in WRITES]
    results.write_text("".join(f"{line}\n" for line in [*kept, *map(json.dumps, rows)]))


if __name__ == "__main__":
    main()
