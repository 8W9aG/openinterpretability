"""Where GPT-2's neurons rest, and what the tokens below the dip have in common.

The numbers behind papers/default-on-poles/README.md. ``scripts/default_on_clamps.py``
measures the interventions. Commands:

* ``rest``: each layer's resting pre-activation, default-on share and GELU slope, trained
  and at random init. It also measures the gain on input noise, with each neuron's rest
  moved to zero as the control.
* ``bands``: the split-half reliability by token of each neuron's tokens below the dip,
  and of random directions of the MLP's input. It also writes each neuron's top tokens.

``scripts/default_on_decode.py`` reads each side with a ridge readout. Windows do not
overlap, and fit and test windows come from different documents.

    uv run python scripts/default_on_poles.py rest --out papers/default-on-poles/results.jsonl
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import click
import torch

from openinterp.cli import _load, resolve_device
from openinterp.cli_evidence import read_corpus
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS, sample_tokens
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.wiring import _project

GPT2 = "openai-community/gpt2"
WINDOW_TOKENS = 128
FIT_WINDOWS = 300
TEST_WINDOWS = 400
BATCH_WINDOWS = 32
SPLIT_SEED = 0
# The share of a pole's maximum that counts as active (docs/PROTOCOL.md §1).
ACTIVE_LINE = 0.25
MIN_TOKEN_COUNT = 10  # occurrences in each half for a token to enter a reliability
RANDOM_INIT_SEEDS = (0, 1, 2, 3, 4)
RANDOM_DIRECTIONS = 200
READ_DIRECTIONS = 5  # random directions whose top tokens are written out to read
NOISE_SCALE = 0.1  # of each input dimension's SD over tokens
NOISE_TOKENS = 20_000
TOP_TOKENS = 25
STUDIED_LAYER = 9
STUDIED_NEURON = 1046
CONTROLS = 10
CONTROL_SEED = 0


def gelu_dip() -> float:
    """Return the pre-activation at GELU's minimum: the root of Φ(x) + xφ(x) on (−1, −0.5)."""
    low, high = -1.0, -0.5
    for _ in range(60):
        middle = (low + high) / 2
        if gelu_slope(torch.tensor(middle, dtype=torch.float64)) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def gelu_slope(h: torch.Tensor) -> torch.Tensor:
    """GELU's derivative: Φ(h) + hφ(h)."""
    phi = torch.exp(-h.square() / 2) / math.sqrt(2 * math.pi)
    return 0.5 * (1 + torch.erf(h / math.sqrt(2))) + h * phi


def write_row(out: Path, row: dict) -> None:
    """Append one JSON line to ``out``, at once, so a long run shows its progress."""
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def batched(ids: torch.Tensor, size: int = BATCH_WINDOWS):
    """Yield ``ids`` in batches of ``size`` windows."""
    yield from ids.split(size)


@dataclass
class Split:
    """Every window of the sample, and the fit and test windows drawn from it."""

    ids: torch.Tensor  # [windows, WINDOW_TOKENS]
    docs: torch.Tensor  # [windows]
    fit: torch.Tensor
    test: torch.Tensor
    test_docs: torch.Tensor
    sample: str


def cut_windows(tokenizer, texts) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each document's whole windows, which do not overlap, and their documents."""
    windows, docs = [], []
    for doc, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(ids) - WINDOW_TOKENS + 1, WINDOW_TOKENS):
            windows.append(ids[start : start + WINDOW_TOKENS])
            docs.append(doc)
    return torch.tensor(windows), torch.tensor(docs)


def split_windows(docs: torch.Tensor, fit: int, test: int, seed: int):
    """Return the indices of fit and test windows, with no document in both.

    Windows of one document share its topic, so a split by window would leak.
    """
    order = torch.randperm(int(docs.max()) + 1, generator=torch.Generator().manual_seed(seed))
    fit_rows: list[int] = []
    test_rows: list[int] = []
    for doc in order.tolist():
        rows = torch.where(docs == doc)[0].tolist()
        (fit_rows if len(fit_rows) < fit else test_rows).extend(rows)
        if len(test_rows) >= test:
            break
    if len(test_rows) < test:
        raise click.ClickException(f"only {len(test_rows)} test windows, wanted {test}")
    return torch.tensor(fit_rows[:fit]), torch.tensor(test_rows[:test])


def load_split(tokenizer, corpus: Path, max_tokens: int, sample_seed: int) -> Split:
    """Read the corpus sample and cut its windows."""
    texts, *_ = read_corpus(corpus)
    texts, sample = sample_tokens(tokenizer, texts, max_tokens, sample_seed)
    ids, docs = cut_windows(tokenizer, texts)
    fit, test = split_windows(docs, FIT_WINDOWS, TEST_WINDOWS, SPLIT_SEED)
    return Split(ids, docs, ids[fit], ids[test], docs[test], sample.sha256[:12])


@torch.no_grad()
def read_pre_activations(runner, ids: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """Return ``{layer: [tokens, width]}``: GPT-2's ``c_fc`` output, first tokens left out."""
    rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []
    for layer in layers:

        def record(_module, _args, out, layer=layer):
            rows[layer].append(out[:, 1:].reshape(-1, out.shape[-1]).float().cpu())

        handles.append(runner.mlp(layer).c_fc.register_forward_hook(record))
    try:
        for batch in batched(ids):
            runner.forward({"input_ids": batch.to(runner.device)})
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(parts) for layer, parts in rows.items()}


def rest_row(h: torch.Tensor, c_fc) -> dict:
    """One layer's resting statistics, from its pre-activations ``h``."""
    median = h.median(0).values
    strength = (-torch.nn.functional.gelu(h)).clamp(min=0)
    active = (strength >= ACTIVE_LINE * strength.max(0).values).float().mean(0)
    slope = gelu_slope(h)
    bias = c_fc.bias.detach().float().cpu()
    return {
        "median_h": median.median().item(),
        "default_on": (active >= 0.5).float().mean().item(),
        "mean_slope": slope.mean().item(),
        "mean_abs_slope": slope.abs().mean().item(),
        "median_bias": bias.median().item(),
        "median_input_term": (median - bias).median().item(),
        "median_h_sd": h.std(0).median().item(),
    }


def random_init_runner(model_name: str, tokenizer, seed: int, device: str) -> NeuronHookRunner:
    """Return the architecture of ``model_name`` with fresh weights from ``seed``."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
    return NeuronHookRunner(model.to(device).eval(), tokenizer)


def rest_rows(runner, ids: torch.Tensor, weights: str) -> list[dict]:
    """Return a row per layer of ``runner``."""
    layers = range(runner.n_layers)
    pre = read_pre_activations(runner, ids, layers)
    return [
        {"part": "rest", "weights": weights, "layer": la, **rest_row(pre[la], runner.mlp(la).c_fc)}
        for la in layers
    ]


@torch.no_grad()
def read_mlp_inputs(runner, ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Return ``[tokens, d_model]``: what ``layer``'s MLP reads, first tokens left out."""
    rows = []

    def record(_module, args):
        rows.append(args[0][:, 1:].reshape(-1, args[0].shape[-1]).cpu())

    handle = runner.mlp(layer).register_forward_pre_hook(record)
    try:
        for batch in batched(ids):
            runner.forward({"input_ids": batch.to(runner.device)})
    finally:
        handle.remove()
    return torch.cat(rows)


def _gains(mlp, x: torch.Tensor, noises: dict, shift: torch.Tensor) -> dict:
    """Return the gain on token-to-token signal and on each noise, with ``shift`` added to h."""

    def write(inputs: torch.Tensor) -> torch.Tensor:
        return _project(mlp.c_proj, torch.nn.functional.gelu(mlp.c_fc(inputs) + shift))

    clean = write(x)
    out = {"signal": clean.var(0).sum().item() / x.var(0).sum().item()}
    for name, noise in noises.items():
        change = write(x + noise) - clean
        out[name] = change.square().sum(1).mean().item() / noise.square().sum(1).mean().item()
    return out


@torch.no_grad()
def noise_row(runner, ids: torch.Tensor, layer: int) -> dict:
    """Compare the layer's MLP with the same MLP with each neuron's median h moved to zero.

    Isotropic noise goes in every direction. The other noise points towards other tokens'
    inputs, so it has the data's own shape. If a resting neuron rejects noise, the signal's
    gain over the noise's gain is higher as trained than recentred.
    """
    mlp = runner.mlp(layer)
    x = read_mlp_inputs(runner, ids, layer)[:NOISE_TOKENS].to(runner.device)
    generator = torch.Generator().manual_seed(layer)
    sd = x.std(0)
    other = x[torch.randperm(len(x), generator=generator).to(x.device)]
    noises = {
        "isotropic": NOISE_SCALE * sd * torch.randn(x.shape, generator=generator).to(x.device),
        "data_shaped": NOISE_SCALE * (other - x),
    }
    median = mlp.c_fc(x).median(0).values
    return {
        "part": "noise",
        "layer": layer,
        "trained": _gains(mlp, x, noises, torch.zeros_like(median)),
        "recentred": _gains(mlp, x, noises, -median),
    }


def split_half(hits: torch.Tensor, tokens: torch.Tensor, even: torch.Tensor) -> torch.Tensor:
    """Return, per column of ``hits``, the correlation by token between two halves of the rows.

    ``hits`` is ``[n, columns]`` and ``even`` marks one half. A token enters if it occurs at
    least ``MIN_TOKEN_COUNT`` times in each half. A value near 1 says the token alone
    predicts the hit.
    """
    types, index = tokens.unique(return_inverse=True)
    rates, counts = [], []
    for half in (even, ~even):
        count = torch.zeros(len(types)).index_add_(0, index[half], torch.ones(int(half.sum())))
        total = torch.zeros(len(types), hits.shape[1])
        rates.append(total.index_add_(0, index[half], hits[half].float()))
        counts.append(count)
    keep = (counts[0] >= MIN_TOKEN_COUNT) & (counts[1] >= MIN_TOKEN_COUNT)
    a = rates[0][keep] / counts[0][keep].unsqueeze(1)
    b = rates[1][keep] / counts[1][keep].unsqueeze(1)
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)


def top_tokens(tokenizer, hit: torch.Tensor, tokens: torch.Tensor) -> list[dict]:
    """Return the tokens with the highest hit rate, each with its rate and count."""
    types, index = tokens.unique(return_inverse=True)
    count = torch.zeros(len(types)).index_add_(0, index, torch.ones(len(tokens)))
    total = torch.zeros(len(types)).index_add_(0, index, hit.float())
    rate = (total / count.clamp(min=1)) * (count >= MIN_TOKEN_COUNT)
    best = rate.topk(TOP_TOKENS).indices
    return [
        {
            "token": tokenizer.decode([types[i].item()]),
            "rate": rate[i].item(),
            "count": int(count[i]),
        }
        for i in best
    ]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the rank correlation of two equal-length vectors (ties broken by position)."""
    rank = lambda v: v.argsort().argsort().float()  # noqa: E731
    return torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1].item()


@torch.no_grad()
def read_surprisal(runner, ids: torch.Tensor) -> torch.Tensor:
    """Return each token's −log p given its prefix, first tokens left out."""
    rows = []
    for batch in batched(ids):
        batch = batch.to(runner.device)
        log_probs = runner.forward({"input_ids": batch})[:, :-1].log_softmax(-1)
        rows.append(-log_probs.gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1).reshape(-1).cpu())
    return torch.cat(rows)


def control_neurons(default_on: torch.Tensor) -> list[int]:
    """Draw the controls at random from the layer's default-on neurons, not the studied one."""
    pool = torch.where(default_on)[0]
    pool = pool[pool != STUDIED_NEURON]
    picks = torch.randperm(len(pool), generator=torch.Generator().manual_seed(CONTROL_SEED))
    return sorted(pool[picks[:CONTROLS]].tolist())


def random_direction_hits(x: torch.Tensor, share: torch.Tensor) -> torch.Tensor:
    """Return ``[n, directions]``: each random direction's lowest tail of ``x``.

    Each direction draws its tail's share from ``share``, the layer's neurons, so the null
    has the neurons' sparsity. A reliability this high needs no neuron at all.
    """
    generator = torch.Generator().manual_seed(0)
    directions = torch.randn(x.shape[1], RANDOM_DIRECTIONS, generator=generator)
    picks = torch.randint(len(share), (RANDOM_DIRECTIONS,), generator=generator)
    values = x @ (directions / directions.norm(dim=0))
    hits = [
        values[:, i] <= values[:, i].quantile(share[picks[i]].clamp(1e-4, 0.9999))
        for i in range(RANDOM_DIRECTIONS)
    ]
    return torch.stack(hits, 1)


def band_rows(runner, split: Split) -> list[dict]:  # long-function: one table, read in order
    """Return the below-dip rows of the studied layer: population, null and read neurons."""
    layer = STUDIED_LAYER
    dip = gelu_dip()
    h = read_pre_activations(runner, split.ids, [layer])[layer]
    x = read_mlp_inputs(runner, split.ids, layer)
    tokens = split.ids[:, 1:].reshape(-1)
    docs = split.docs.repeat_interleave(WINDOW_TOKENS - 1)
    strength = (-torch.nn.functional.gelu(h)).clamp(min=0)
    scale = strength.max(0).values
    default_on = (strength >= ACTIVE_LINE * scale).float().mean(0) >= 0.5
    # Below the dip and below the active line: GELU has climbed back towards zero.
    below = (h < dip) & (strength < ACTIVE_LINE * scale)
    even = docs % 2 == 0
    reliability = split_half(below, tokens, even)
    share = below.float().mean(0)
    matched = random_direction_hits(x, share[STUDIED_NEURON].unsqueeze(0))
    population = reliability[default_on]
    rows = [
        {
            "part": "band_population",
            "layer": layer,
            "dip": dip,
            "default_on_neurons": int(default_on.sum()),
            "median_below_share": share[default_on].median().item(),
            "reliability_quantiles": _quantiles(population),
            "studied_percentile": (population < reliability[STUDIED_NEURON]).float().mean().item(),
            "random_direction_quantiles": _quantiles(
                split_half(random_direction_hits(x, share[default_on]), tokens, even)
            ),
            "random_direction_at_studied_share": _quantiles(split_half(matched, tokens, even)),
        }
    ]
    rows += [
        {
            "part": "band_random_direction",
            "layer": layer,
            "direction": i,
            "share": matched[:, i].float().mean().item(),
            "top_tokens": top_tokens(runner.tokenizer, matched[:, i], tokens),
        }
        for i in range(READ_DIRECTIONS)
    ]
    by_occurrence = split_half(
        below, tokens, torch.rand(len(tokens), generator=torch.Generator().manual_seed(0)) < 0.5
    )
    surprisal = read_surprisal(runner, split.ids)
    for neuron in [STUDIED_NEURON, *control_neurons(default_on)]:
        row = _neuron_row(runner, neuron, h[:, neuron], below[:, neuron], tokens, surprisal)
        row["reliability"] = reliability[neuron].item()
        row["reliability_by_occurrence"] = by_occurrence[neuron].item()
        rows.append(row)
    return rows


def _neuron_row(runner, neuron, h, below, tokens, surprisal) -> dict:
    """One read neuron's share below the dip, its link to surprisal, and its top tokens."""
    return {
        "part": "band_neuron",
        "layer": STUDIED_LAYER,
        "neuron": neuron,
        "below_share": below.float().mean().item(),
        "share_h_below_dip": (h < gelu_dip()).float().mean().item(),
        "median_h": h.median().item(),
        "surprisal_rho": spearman(h, surprisal),
        "below_mean_surprisal": surprisal[below].mean().item(),
        "other_mean_surprisal": surprisal[~below].mean().item(),
        "top_tokens": top_tokens(runner.tokenizer, below, tokens),
    }


def _quantiles(values: torch.Tensor) -> dict:
    """The 5th, 50th and 95th percentiles of ``values``."""
    points = torch.tensor([0.05, 0.5, 0.95])
    low, median, high = values[~values.isnan()].quantile(points).tolist()
    return {"q05": low, "q50": median, "q95": high}


@click.group()
def main():
    """Measure default-on negative poles as a population."""


def _setup(device: str, corpus: Path, max_tokens: int, sample_seed: int):
    runner = _load(GPT2, resolve_device(device), torch.float32, None, None)
    return runner, load_split(runner.tokenizer, corpus, max_tokens, sample_seed)


def _options(fn):
    for option in (
        click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl")),
        click.option("--max-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS, show_default=True),
        click.option("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, show_default=True),
        click.option("--device", default="auto", show_default=True),
        click.option("--out", type=click.Path(path_type=Path), required=True),
    ):
        fn = option(fn)
    return fn


@main.command()
@_options
def rest(corpus, max_tokens, sample_seed, device, out):
    """Write the resting statistics, trained and at random init, and the noise gains."""
    runner, split = _setup(device, corpus, max_tokens, sample_seed)
    for row in rest_rows(runner, split.test, "trained"):
        write_row(out, {"model": GPT2, "sample": split.sample, **row})
    for layer in range(runner.n_layers):
        row = noise_row(runner, split.test, layer)
        write_row(out, {"model": GPT2, "sample": split.sample, **row})
    for seed in RANDOM_INIT_SEEDS:
        fresh = random_init_runner(GPT2, runner.tokenizer, seed, runner.device)
        for row in rest_rows(fresh, split.test, f"init_{seed}"):
            write_row(out, {"model": GPT2, "sample": split.sample, **row})
        click.echo(f"random init {seed} done", err=True)


@main.command()
@_options
def bands(corpus, max_tokens, sample_seed, device, out):
    """Write the below-dip rows of the studied layer."""
    runner, split = _setup(device, corpus, max_tokens, sample_seed)
    for row in band_rows(runner, split):
        write_row(out, {"model": GPT2, "sample": split.sample, **row})


if __name__ == "__main__":
    main()
