"""What the model loses when one side of its MLP activations stops varying.

The numbers behind papers/default-on-poles/README.md. Constants come from the fit windows.
Each condition runs on test windows from other documents, at the input of the down
projection. A *side* of a neuron is its tokens with activation below zero (negative) or
above zero (positive). Conditions:

* ``clamp``: each neuron holds its side's mean on that side's tokens.
* ``zero``: each neuron reads zero on that side's tokens.
* ``mean``: each neuron holds its mean on every token.
* ``null``: each neuron gets the clamp's change, shuffled over the batch's tokens. The
  size of the change per neuron stays, and its link to the token goes.
* ``frozen``: the clamp at one layer, with later MLP writes kept at their clean values.
* ``direct``: the clamp's change to one layer's write, added at the final norm only.

The gap between ``clamp`` and ``direct`` shows how much the later layers compensate. Each
condition leaves a window's first token as it is. That token has massive activations and
is the attention sink (Sun et al., 2024, arXiv:2402.17762). The ``_with_first`` conditions
change it too, to measure what this costs.

    uv run python scripts/default_on_clamps.py --model openai-community/gpt2 --device auto \\
        --out papers/default-on-poles/results.jsonl
"""

from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import STUDIED_LAYER, batched, load_split, write_row

from openinterp.cli import _load, resolve_device
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS
from openinterp.harness.hooks import ablate_pole
from openinterp.harness.wiring import _project

SIDES = ("negative", "positive")
BOOTSTRAPS = 1000


def on_side(acts: torch.Tensor, side: str) -> torch.Tensor:
    """Return where ``acts`` lie on ``side`` of zero."""
    return acts < 0 if side == "negative" else acts > 0


@torch.no_grad()
def fit_constants(runner, ids: torch.Tensor) -> dict:
    """Return each layer's neuron means: per side, and over all tokens, first tokens left out."""
    totals: dict = defaultdict(lambda: 0.0)

    def recorder(layer: int):
        def record(acts: torch.Tensor) -> None:
            rest = acts[:, 1:].reshape(-1, acts.shape[-1]).float()
            for side in SIDES:
                mask = on_side(rest, side)
                totals[layer, side, "sum"] += (rest * mask).sum(0).cpu()
                totals[layer, side, "count"] += mask.sum(0).cpu()
            totals[layer, "mean", "sum"] += rest.sum(0).cpu()
            totals[layer, "mean", "count"] += rest.shape[0]
            totals[layer, "negative", "tokens"] += on_side(rest, "negative").float().mean().item()
            totals[layer, "negative", "batches"] += 1

        return record

    recorders = {layer: recorder(layer) for layer in range(runner.n_layers)}
    for batch in batched(ids):
        with runner.hooks(recorders):
            runner.forward({"input_ids": batch.to(runner.device)})
    shares = {
        (layer, "negative_share"): totals[layer, "negative", "tokens"]
        / totals[layer, "negative", "batches"]
        for layer in range(runner.n_layers)
    }
    return shares | {
        (layer, part): totals[layer, part, "sum"]
        / torch.as_tensor(totals[layer, part, "count"]).clamp(min=1)
        for layer in range(runner.n_layers)
        for part in (*SIDES, "mean")
    }


def keep_first(intervene):
    """Return ``intervene`` with each window's first token left as it is."""

    def kept(acts: torch.Tensor) -> torch.Tensor:
        out = intervene(acts).clone()
        out[:, 0] = acts[:, 0]
        return out

    return kept


def clamp(value: torch.Tensor, side: str):
    """An intervention that holds ``side``'s tokens at ``value``, per neuron."""

    def intervene(acts: torch.Tensor) -> torch.Tensor:
        return torch.where(on_side(acts, side), value.to(acts), acts)

    return intervene


def zero(side: str):
    """An intervention that sets ``side``'s tokens to zero."""

    def intervene(acts: torch.Tensor) -> torch.Tensor:
        return torch.where(on_side(acts, side), torch.zeros_like(acts), acts)

    return intervene


def hold_mean(value: torch.Tensor):
    """An intervention that holds every token at ``value``, per neuron."""

    def intervene(acts: torch.Tensor) -> torch.Tensor:
        return value.to(acts).expand_as(acts)

    return intervene


def shuffle_within_neurons(change: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return ``[b, seq, width]`` ``change``, with each neuron's tokens in its own random order."""
    flat = change.reshape(-1, change.shape[-1])
    order = torch.rand(flat.shape, generator=generator).argsort(0).to(flat.device)
    return flat.gather(0, order).reshape(change.shape)


def shuffled_clamp(value: torch.Tensor, side: str, generator: torch.Generator):
    """The null for :func:`clamp`: its change, moved to other tokens within each neuron.

    First tokens neither give nor receive a change, as in :func:`keep_first`.
    """
    apply = clamp(value, side)

    def intervene(acts: torch.Tensor) -> torch.Tensor:
        out = acts.clone()
        out[:, 1:] += shuffle_within_neurons((apply(acts) - acts)[:, 1:], generator)
        return out

    return intervene


@contextmanager
def added_at_final_norm(runner, layer: int, intervene):
    """Add ``intervene``'s change to ``layer``'s write at the final norm, and nowhere else."""
    change = {}
    proj = runner.down_proj(layer)

    def read(_module, args):
        # _project skips the module's hooks, so this hook does not call itself.
        change["write"] = _project(proj, intervene(args[0])) - _project(proj, args[0])

    def add(_module, args):
        return (args[0] + change["write"], *args[1:])

    handles = [
        proj.register_forward_pre_hook(read),
        runner.final_norm().register_forward_pre_hook(add),
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def read_writes(runner, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Return the clean logits and each layer's MLP write."""
    writes = {}

    def recorder(layer: int):
        def record(out: torch.Tensor) -> None:
            writes[layer] = out.detach().clone()

        return record

    recorders = {layer: recorder(layer) for layer in range(runner.n_layers)}
    with runner.hooks(output_interventions=recorders):
        logits = runner.forward({"input_ids": batch})
    return logits, writes


def score(clean: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Return per window: mean KL(clean‖changed), cross-entropy and top-prediction agreement."""
    log_clean = clean.log_softmax(-1)
    log_new = logits.log_softmax(-1)
    return {
        "kl": (log_clean.exp() * (log_clean - log_new)).sum(-1).mean(1).cpu(),
        "ce": -log_new.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(1).cpu(),
        "same_top": (clean.argmax(-1) == logits.argmax(-1)).float().mean(1).cpu(),
    }


def all_layer_conditions(constants: dict, n_layers: int, generator) -> dict:
    """Name -> ``{layer: intervention}``, for conditions on every layer at once."""
    layers = range(n_layers)
    changes = {"mean": lambda la: hold_mean(constants[la, "mean"])}
    for side in SIDES:
        changes[f"clamp_{side}"] = lambda la, side=side: clamp(constants[la, side], side)
        changes[f"zero_{side}"] = lambda la, side=side: zero(side)
    conditions = {}
    for name, change in changes.items():
        conditions[name] = {la: keep_first(change(la)) for la in layers}
        conditions[f"{name}_with_first"] = {la: change(la) for la in layers}
    for side in SIDES:
        conditions[f"null_{side}"] = {
            la: shuffled_clamp(constants[la, side], side, generator) for la in layers
        }
    return conditions


def single_neuron_conditions(neurons: list[int]) -> dict:
    """Name -> intervention that zeroes one neuron's negative side at the studied layer."""
    return {f"N{n}": keep_first(ablate_pole(n, "negative")) for n in neurons}


class Tally:
    """Per-window scores, and the sizes of the changes, for every condition."""

    def __init__(self) -> None:
        self.rows: dict = defaultdict(lambda: defaultdict(list))
        self.size: dict = defaultdict(lambda: 0.0)
        self.tokens: dict = defaultdict(int)

    def add(self, key: tuple, scores: dict) -> None:
        """Keep one batch's per-window scores under ``key``."""
        for name, values in scores.items():
            self.rows[key][name].append(values)

    def add_size(self, key: tuple, proj, change: torch.Tensor) -> None:
        """Add one batch's squared change, in activations and in the layer's write."""
        flat = change.reshape(-1, change.shape[-1])
        self.size[(*key, "act")] += flat.square().sum().item()
        self.size[(*key, "write")] += (
            _project(proj, flat)
            .sub(_project(proj, torch.zeros_like(flat[:1])))
            .square()
            .sum()
            .item()
        )
        self.tokens[key] += flat.shape[0]

    def values(self, key: tuple, name: str = "kl") -> torch.Tensor:
        """``[windows]``: one measure of one condition."""
        return torch.cat(self.rows[key][name])


def run_batch(runner, batch, constants, tally: Tally, generator, neurons, layers) -> None:
    """Add every condition's scores on one batch of test windows."""
    inputs = {"input_ids": batch}
    targets = batch[:, 1:]
    clean, writes = read_writes(runner, batch)
    clean = clean[:, :-1]
    tally.add(("clean",), score(clean, clean, targets))

    def measure(key, interventions=None, output_interventions=None):
        logits = runner.run_with_interventions(inputs, interventions, output_interventions)
        tally.add(key, score(clean, logits[:, :-1], targets))

    n_layers = runner.n_layers
    for name, interventions in all_layer_conditions(constants, n_layers, generator).items():
        measure(("all", name), interventions)
    for name, intervene in single_neuron_conditions(neurons).items():
        measure(("single", name), {STUDIED_LAYER: intervene})
    for layer in layers:
        _run_layer(runner, batch, constants, tally, generator, layer, clean, targets, writes)


def _run_layer(runner, batch, constants, tally, generator, layer, clean, targets, writes) -> None:
    """Add the conditions on one layer, for each side."""
    inputs = {"input_ids": batch}
    frozen = {
        later: (lambda out, kept=writes[later]: kept) for later in range(layer + 1, runner.n_layers)
    }
    for side in SIDES:
        apply = keep_first(clamp(constants[layer, side], side))

        def sizes(acts, key=("layer", layer, side), apply=apply):
            tally.add_size(key, runner.down_proj(layer), apply(acts) - acts)
            return apply(acts)

        changes = {
            "clamp": ({layer: sizes}, None),
            "null": ({layer: shuffled_clamp(constants[layer, side], side, generator)}, None),
            "frozen": ({layer: apply}, frozen),
        }
        for name, (interventions, outputs) in changes.items():
            logits = runner.run_with_interventions(inputs, interventions, outputs)
            tally.add(("layer", layer, side, name), score(clean, logits[:, :-1], targets))
        with added_at_final_norm(runner, layer, apply), torch.no_grad():
            logits = runner.forward(inputs)
        tally.add(("layer", layer, side, "direct"), score(clean, logits[:, :-1], targets))


def ratio_interval(top: torch.Tensor, bottom: torch.Tensor, docs: torch.Tensor) -> list[float]:
    """Return the 95% interval of ``top``'s mean over ``bottom``'s, resampling documents."""
    unique = docs.unique()
    index = [torch.where(docs == d)[0] for d in unique]
    generator = torch.Generator().manual_seed(0)
    ratios = []
    for _ in range(BOOTSTRAPS):
        picks = torch.randint(len(unique), (len(unique),), generator=generator)
        rows = torch.cat([index[i] for i in picks.tolist()])
        ratios.append((top[rows].mean() / bottom[rows].mean()).item())
    low, high = torch.tensor(ratios).quantile(torch.tensor([0.025, 0.975])).tolist()
    return [low, high]


def summary_rows(tally: Tally, docs: torch.Tensor, constants: dict, layers) -> list[dict]:
    """Return the JSON rows: whole-model conditions, single neurons, then each layer."""
    base = tally.values(("clean",), "ce").mean().item()
    rows = [{"part": "clean", "ce": base}]
    for key in tally.rows:
        if key[0] in ("all", "single"):
            rows.append(_condition_row(tally, key, base))
    for layer in layers:
        share = {"negative_token_share": constants[layer, "negative_share"]}
        rows.append(_layer_row(tally, layer, docs) | share)
    return rows


def _condition_row(tally: Tally, key: tuple, base: float) -> dict:
    """One whole-model or single-neuron condition, against the clean cross-entropy."""
    return {
        "part": key[0],
        "condition": key[1],
        "kl": tally.values(key).mean().item(),
        "added_ce": tally.values(key, "ce").mean().item() - base,
        "same_top": tally.values(key, "same_top").mean().item(),
    }


def _layer_row(tally: Tally, layer: int, docs: torch.Tensor) -> dict:
    """One layer's KL per side and condition, the sizes of the changes, and the ratios."""
    row: dict = {"part": "layer", "layer": layer}
    for side in SIDES:
        key = ("layer", layer, side)
        tokens = tally.tokens[key]
        row[f"{side}_act_variance"] = tally.size[(*key, "act")] / tokens
        row[f"{side}_write_variance"] = tally.size[(*key, "write")] / tokens
        for name in ("clamp", "null", "frozen", "direct"):
            row[f"{side}_{name}_kl"] = tally.values((*key, name)).mean().item()
    neg = tally.values(("layer", layer, "negative", "clamp"))
    pos = tally.values(("layer", layer, "positive", "clamp"))
    row["kl_ratio"] = (neg.mean() / pos.mean()).item()
    row["kl_ratio_interval"] = ratio_interval(neg, pos, docs)
    for space in ("act", "write"):
        row[f"{space}_variance_ratio"] = (
            row[f"negative_{space}_variance"] / row[f"positive_{space}_variance"]
        )
    return row


@click.command()
@click.option("--model", "model_name", required=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--max-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS, show_default=True)
@click.option("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--neurons", default="", help="Comma-separated neurons of the studied layer.")
@click.option("--layers", "layer_list", default="", help="Comma-separated; default every layer.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(model_name, corpus, max_tokens, sample_seed, device, neurons, layer_list, out):
    """Append the clamp rows of ``model_name`` to ``--out``."""
    runner = _load(model_name, resolve_device(device), torch.float32, None, None)
    split = load_split(runner.tokenizer, corpus, max_tokens, sample_seed)
    constants = fit_constants(runner, split.fit)
    generator = torch.Generator().manual_seed(0)
    picked = [int(n) for n in neurons.split(",") if n]
    layers = [int(n) for n in layer_list.split(",") if n] or list(range(runner.n_layers))
    tally = Tally()
    with torch.no_grad():
        for i, batch in enumerate(batched(split.test)):
            run_batch(runner, batch.to(runner.device), constants, tally, generator, picked, layers)
            click.echo(f"batch {i + 1}/{len(split.test) // 32 + 1} done", err=True)
    for row in summary_rows(tally, split.test_docs, constants, layers):
        write_row(out, {"model": model_name, "sample": split.sample, **row})


if __name__ == "__main__":
    main()
