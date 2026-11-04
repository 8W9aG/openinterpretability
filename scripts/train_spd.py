"""Decompose one model's MLP projections with Stochastic Parameter Decomposition.

This is the parameter-space baseline for the native-neuron premise. SPD is from Bushnaq
et al. (2025, arXiv:2506.20790), after APD (Braun et al., 2025, arXiv:2501.14926). It
writes each down projection ``W`` ``[d_mlp, d_model]`` as a sum of C rank-1
subcomponents ``V[:, c] ⊗ U[c]``. On hidden activations ``h``, a subcomponent's inner
activation is ``h · V[:, c]``. A gate gives each subcomponent a causal importance
``g_c`` in [0, 1] on each token. Training makes the output survive any ablation of each
subcomponent down to ``g_c``. It also makes ``g`` sparse.

The down projection reads the MLP's hidden activations, where native neurons are. So a
subcomponent and a neuron are directions in one space. Neuron i is ``V = e_i, U = W[i]``.

``--matrices up`` or ``both`` also decomposes the MLP's input projection, which writes
the neurons' pre-activations. There, neuron i is ``V = W[:, i], U = e_i``. The
activation function acts on each neuron separately, so it can favour subcomponents that
are neurons. A neuron that is off on a token can be dropped there. A subcomponent that
writes to many neurons can be dropped only where all of them are off. Its CI-masked
output loss and explained variance are measured after the activation function.

The losses follow the paper and Goodfire's reference code
(github.com/goodfire-ai/param-decomp, ``nano_param_decomp/run.py``):

* faithfulness: the subcomponents sum to ``W``;
* stochastic reconstruction: subcomponents are masked by ``g + (1 - g)·U(0, 1)``, and
  the rest ``W - VU`` by ``U(0, 1)``. The output's KL from the model's is small;
* the same, with one random layer masked at a time;
* CI-masked reconstruction: the same KL, with each subcomponent masked by exactly ``g``
  and no delta. The harness reads the decomposition this way. Without this loss, ``g``
  fell to a median of 0.02 on GPT-2;
* CI-masked output: each layer's projection at ``g`` matches the model's in mean squared
  error. With the KL losses alone, GPT-2's masked outputs kept 20-45% of their norm;
* importance minimality: ``(g + eps)^p`` is small. The reference's ``eps = 1e-12`` froze
  GPT-2's gates at ``p = 0.4``. Here ``--imp-eps`` is 1e-3, and ``--gate-grad-clip``
  clips the gate gradients.

The losses are in ``spd_losses.py``. With ``--gate layer``, one GELU MLP per layer reads
the whole centred hidden vector and gives every subcomponent's importance. The paper
has one MLP per subcomponent on its own inner activation (``spd_model.py``).

Each subcomponent reads the hidden activations minus their mean (``--center``). The
mean's write goes to the bias. GELU leaves most hidden units at small negative values.
Without centring, an ablation to zero changes GPT-2's output on every token.

The first position of each piece is the attention sink, whose activations are outliers.
It keeps the model's weights and is left out of every loss.

``--resume`` continues from a checkpoint with a new optimizer. ``--init neurons``
starts with one subcomponent per neuron.

The data is pile-10k without the scan corpus's documents (``train_transcoders.py``).
Each layer goes to ``<out>/layer_<layer>.safetensors``. ``openinterp.harness.dictionaries``
reads it with ``activation="causal_importance"``. Only subcomponents alive on held-out
text are kept. Each becomes two latents, the positive and negative parts of
``g·(h·V[:, c])``. They write ``+U[c]`` and ``-U[c]``. Statistics go to
``<out>/layer_<layer>.json``.

    uv run python scripts/train_spd.py --model openai-community/gpt2 --layers all \\
        --steps 20000 --out exchange/gpt2-spd
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from spd_losses import (
    faithfulness,
    first_kept,
    importance_loss,
    kl,
    lower_leaky,
    output_loss,
    projected,
    stochastic_masks,
    upper_leaky,
)
from spd_model import MATRICES, Decomposer, latent_pairs, load_parts, site_label
from train_transcoders import PILE, VALIDATION_DOCS, sequences, training_texts


@dataclass(frozen=True)
class Settings:
    """``main``'s options."""

    model_name: str
    revision: str | None
    layers: str
    matrices: str
    components: int
    d_gate: int
    gate: str
    init: str
    steps: int
    batch_rows: int
    seq_len: int
    lr: float
    faith_warmup_steps: int
    coeff_faith: float
    coeff_stoch: float
    coeff_layerwise: float
    coeff_ci_masked: float
    coeff_ci_output: float
    coeff_imp: float
    center: bool
    p_start: float
    p_end: float
    p_anneal: float
    imp_beta: float
    imp_eps: float
    leaky_alpha: float
    grad_clip: float
    gate_grad_clip: float
    eval_every: int
    alive_rows: int
    seed: int
    device: str
    scan_corpus: Path
    out: Path
    resume: Path | None


PATHS = ("scan_corpus", "out", "resume")  # left out of the exported settings


def cosine_lr(step: int, steps: int, lr: float, final_frac: float = 0.1) -> float:
    """Cosine decay from ``lr`` to ``lr·final_frac``."""
    progress = min(step / max(1, steps - 1), 1.0)
    return lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def batches(rows: list[list[int]], cfg: Settings, device):
    """Endless ``[batch_rows, seq_len]`` batches, reshuffled each pass."""
    rng = random.Random(cfg.seed)
    while True:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in range(0, len(order) - cfg.batch_rows + 1, cfg.batch_rows):
            yield torch.tensor([rows[j] for j in order[i : i + cfg.batch_rows]], device=device)


def losses(dec: Decomposer, cfg: Settings, ids: torch.Tensor, step: int) -> dict:
    """The six SPD losses on one batch, and the importance loss's exponent ``p``."""
    target, hidden, inner, logits = dec.importance(ids)
    lower = {site: lower_leaky(z, cfg.leaky_alpha) for site, z in logits.items()}
    upper = {site: upper_leaky(z, cfg.leaky_alpha) for site, z in logits.items()}
    progress = min(1.0, step / (cfg.p_anneal * cfg.steps))
    p = cfg.p_start + (cfg.p_end - cfg.p_start) * progress
    every = {site: stochastic_masks(g) for site, g in lower.items()}
    no_delta = first_kept(torch.zeros(ids.shape, device=ids.device), 1.0)
    at_ci = {site: (first_kept(g, 1.0), no_delta) for site, g in lower.items()}
    one = random.Random(cfg.seed * 1_000_003 + step).choice(dec.sites)
    return {
        "faith": faithfulness(dec.parts),
        "imp": importance_loss(upper, p, cfg.imp_beta, cfg.imp_eps),
        "stoch": kl(dec.logits(ids, every), target),
        "layerwise": kl(dec.logits(ids, {one: stochastic_masks(lower[one])}), target),
        "ci_masked": kl(dec.logits(ids, at_ci), target),
        "ci_output": output_loss(dec.parts, hidden, inner, lower, dec.acts),
        "p": p,
    }


def save_checkpoint(cfg: Settings, step: int, parts: dict) -> None:
    """Write ``<out>/checkpoint.pt``. An interrupted write leaves the last one whole."""
    settings = {k: v for k, v in asdict(cfg).items() if k not in PATHS}
    tmp = cfg.out / "checkpoint.pt.tmp"
    torch.save({"step": step, "parts": parts, "settings": settings}, tmp)
    os.replace(tmp, cfg.out / "checkpoint.pt")


def first_step(dec: Decomposer, cfg: Settings) -> int:
    """Load ``cfg.resume``, or fit the subcomponents to the weights; the step to start at.

    A resumed run starts with a new optimizer, which clears Adam's moment estimates.
    """
    if cfg.resume is not None:
        state = torch.load(cfg.resume, map_location=dec.runner.device)
        saved = load_parts(state["parts"])
        for site, part in dec.parts.items():
            part.load_state_dict(saved[site])
        click.echo(f"resumed from step {state['step']} of {cfg.resume}")
        return state["step"] + 1
    components = dec.parameters("components")
    warm = torch.optim.AdamW(components, lr=1e-3, weight_decay=0.0)
    for _ in range(cfg.faith_warmup_steps):
        warm.zero_grad()
        faithfulness(dec.parts).backward()
        warm.step()
    click.echo(f"faithfulness warmup: {float(faithfulness(dec.parts).detach()):.3g}")
    return 0


def train(dec: Decomposer, cfg: Settings, rows, valid: torch.Tensor) -> None:
    """Run the SPD losses from the first step to ``cfg.steps``."""
    first = first_step(dec, cfg)
    components, gates = dec.parameters("components"), dec.parameters("gates")
    opt = torch.optim.AdamW(components + gates, lr=cfg.lr, weight_decay=0.0)
    coeffs = {"faith": cfg.coeff_faith, "imp": cfg.coeff_imp, "stoch": cfg.coeff_stoch,
              "layerwise": cfg.coeff_layerwise, "ci_masked": cfg.coeff_ci_masked,
              "ci_output": cfg.coeff_ci_output}  # fmt: skip
    log = (cfg.out / "train.jsonl").open("a")
    data, start = batches(rows, cfg, dec.runner.device), time.time()
    for _ in range(first):  # the batches a resumed run has already seen
        next(data)
    for step in range(first, cfg.steps):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, cfg.steps, cfg.lr)
        parts = losses(dec, cfg, next(data), step)
        opt.zero_grad(set_to_none=True)
        sum(coeff * parts[name] for name, coeff in coeffs.items()).backward()
        torch.nn.utils.clip_grad_norm_(components, cfg.grad_clip)
        parts["gate_grad"] = torch.nn.utils.clip_grad_norm_(gates, cfg.gate_grad_clip)
        opt.step()
        row = {"step": step, **{k: float(v) for k, v in parts.items()},
               "seconds": round(time.time() - start, 1)}  # fmt: skip
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            row |= evaluate(dec, valid)
            hours = (cfg.steps - step) * (time.time() - start) / (step + 1 - first) / 3600
            click.echo(f"step {step}: {json.dumps(row)}  ({hours:.1f} h left)")
            state = {site: part.state_dict() for site, part in dec.parts.items()}
            save_checkpoint(cfg, step, state)
        log.write(json.dumps(row) + "\n")
        log.flush()


@torch.no_grad()
def evaluate(dec: Decomposer, ids: torch.Tensor) -> dict:
    """Fit on held-out pieces, leaving out each piece's first position.

    ``kl_ci``: every site at its causal importances, no delta. ``ce_unrecovered``: the
    share of the loss rise from zero-ablating the decomposed projections (to their bias)
    that the CI-masked model keeps. ``explained_variance`` and ``l0`` are per site: one
    site's output from the model's own inputs, as the harness measures it. An input
    projection's output is measured after the activation function.
    """
    target, hidden, inner, logits = dec.importance(ids)
    ci = {site: z.clamp(0, 1) for site, z in logits.items()}
    zero = torch.zeros(ids.shape, device=ids.device)

    def ci_masks(site):
        return first_kept(ci[site], 1.0), first_kept(zero, 1.0)

    def ablated(site):
        return first_kept(torch.zeros_like(ci[site]), 1.0), first_kept(zero, 1.0)

    masked = dec.logits(ids, {site: ci_masks(site) for site in dec.sites})
    none = dec.logits(ids, {site: ablated(site) for site in dec.sites})

    def ce(logits_):
        return float(F.cross_entropy(logits_[:, 1:-1].flatten(0, 1), ids[:, 2:].flatten()))

    ce_target, ce_ci, ce_zero = ce(target), ce(masked), ce(none)
    variance, l0 = {}, {}
    for site, part in dec.parts.items():
        y, y_hat = projected(part, hidden[site][:, 1:], inner[site][:, 1:], ci[site][:, 1:],
                             dec.acts.get(site))  # fmt: skip
        spread = (y - y.mean((0, 1))).pow(2).sum()
        variance[site_label(site)] = round(1 - float((y - y_hat).pow(2).sum() / spread), 4)
        l0[site_label(site)] = round(float((ci[site][:, 1:] > 0).float().sum(-1).mean()), 1)
    return {
        "kl_ci": round(float(kl(masked, target)), 5),
        "ce_target": round(ce_target, 4),
        "ce_ci": round(ce_ci, 4),
        "ce_unrecovered": round((ce_ci - ce_target) / (ce_zero - ce_target), 4),
        "explained_variance": variance,
        "l0": l0,
    }


@torch.no_grad()
def alive(dec: Decomposer, cfg: Settings, ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per site, the subcomponents with a causal importance above 0 on some token of ``ids``."""
    fired = {site: torch.zeros(cfg.components, dtype=torch.bool, device=ids.device)
             for site in dec.sites}  # fmt: skip
    for i in range(0, ids.shape[0], cfg.batch_rows):
        *_, logits = dec.importance(ids[i : i + cfg.batch_rows])
        for site, z in logits.items():
            fired[site] |= (z[:, 1:] > 0).flatten(0, 1).any(0)
    return {site: f.nonzero().squeeze(-1) for site, f in fired.items()}


@torch.no_grad()
def export(dec: Decomposer, cfg: Settings, alive_ids: torch.Tensor, stats: dict) -> None:
    """Write each site's alive subcomponents as latent pairs, with its statistics.

    A down projection goes to ``layer_<layer>``, an input projection to ``layer_<layer>.up``.
    """
    from safetensors.torch import save_file

    kept = alive(dec, cfg, alive_ids)
    for site, part in dec.parts.items():
        keep, label = kept[site], site_label(site)
        save_file(latent_pairs(part, keep), cfg.out / f"layer_{label}.safetensors")
        layer_stats = {
            **stats,
            "layer": site[0],
            "input_site": "mlp_hidden" if site[1] == "down" else "mlp_in",
            "alive": len(keep),
            "alive_tokens": int(alive_ids[:, 1:].numel()),
            "subcomponents": keep.tolist(),
            "explained_variance": stats["final"]["explained_variance"][label],
            "l0": stats["final"]["l0"][label],
        }
        (cfg.out / f"layer_{label}.json").write_text(json.dumps(layer_stats, indent=2) + "\n")
        click.echo(f"L{label}: {len(keep)} of {cfg.components} subcomponents alive")


def pieces(tokenizer, cfg: Settings) -> tuple[list, list, int]:
    """Training and held-out pieces of ``seq_len`` tokens, and the scan documents left out."""
    train_texts, valid_texts, excluded = training_texts(cfg.scan_corpus)
    rows, valid = (
        [r for r in sequences(tokenizer, texts, cfg.seq_len, cfg.seed) if len(r) == cfg.seq_len]
        for texts in (train_texts, valid_texts)
    )
    click.echo(f"{len(rows):,} training pieces of {cfg.seq_len} tokens; "
               f"{len(valid)} held out; {excluded} scan documents left out")  # fmt: skip
    return rows, valid, excluded


@click.command()
@click.option("--model", "model_name", default="openai-community/gpt2", show_default=True)
@click.option("--revision", default=None, help="Model commit (default: the latest).")
@click.option("--layers", required=True, help="Comma-separated layers, or 'all'.")
@click.option(
    "--matrices",
    type=click.Choice(list(MATRICES)),
    default="down",
    show_default=True,
    help="Decompose the MLP's down projection, its input projection, or both.",
)
@click.option("--components", type=int, default=3072, show_default=True, help="C per matrix.")
@click.option(
    "--gate",
    type=click.Choice(["component", "layer"]),
    default="component",
    show_default=True,
    help="One gate per subcomponent (the paper's), or per layer.",
)
@click.option(
    "--init",
    type=click.Choice(["random", "neurons"]),
    default="random",
    show_default=True,
    help="Random subcomponents, or one per neuron (needs C = d_mlp).",
)
@click.option("--d-gate", type=int, default=8, show_default=True, help="The gate's hidden width.")
@click.option("--steps", type=int, default=20000, show_default=True)
@click.option("--batch-rows", type=int, default=8, show_default=True)
@click.option("--seq-len", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--faith-warmup-steps", type=int, default=400, show_default=True)
@click.option("--coeff-faith", type=float, default=1e7, show_default=True)
@click.option("--coeff-stoch", type=float, default=0.5, show_default=True)
@click.option("--coeff-layerwise", type=float, default=0.5, show_default=True)
@click.option("--coeff-ci-masked", type=float, default=1.0, show_default=True)
@click.option("--coeff-ci-output", type=float, default=1.0, show_default=True)
@click.option("--coeff-imp", type=float, default=2e-5, show_default=True)
@click.option("--center/--no-center", default=True, show_default=True,
              help="Read hidden activations relative to their mean.")  # fmt: skip
@click.option("--p-start", type=float, default=2.0, show_default=True)
@click.option("--p-end", type=float, default=0.4, show_default=True)
@click.option("--p-anneal", type=float, default=1.0, show_default=True,
              help="Share of the steps over which p falls to --p-end, then stays.")  # fmt: skip
@click.option("--imp-beta", type=float, default=0.5, show_default=True)
@click.option("--imp-eps", type=float, default=1e-3, show_default=True,
              help="Added to g in the importance loss; bounds its slope at g = 0.")  # fmt: skip
@click.option("--leaky-alpha", type=float, default=0.01, show_default=True)
@click.option("--grad-clip", type=float, default=0.01, show_default=True,
              help="Gradient-norm clip on the subcomponents.")  # fmt: skip
@click.option("--gate-grad-clip", type=float, default=1.0, show_default=True,
              help="Gradient-norm clip on the gates.")  # fmt: skip
@click.option("--eval-every", type=int, default=500, show_default=True)
@click.option("--alive-rows", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--scan-corpus", type=click.Path(exists=True, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--resume", type=click.Path(exists=True, path_type=Path), default=None,
              help="A checkpoint.pt to continue from, with a new optimizer.")  # fmt: skip
def main(**options):
    """Decompose the MLP projections and write each site's latents and statistics."""
    from openinterp.cli import _load_runner

    cfg = Settings(**options)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # on CUDA, ~3x faster float32 matmuls
    cfg.out.mkdir(parents=True, exist_ok=True)
    runner = _load_runner(cfg.model_name, cfg.device, "float32", cfg.revision)
    runner.model.requires_grad_(False)
    layers = range(runner.n_layers) if cfg.layers == "all" else map(int, cfg.layers.split(","))
    layers = list(layers)
    rows, valid_rows, excluded = pieces(runner.tokenizer, cfg)
    device = runner.device
    eval_ids = torch.tensor(valid_rows[: cfg.batch_rows], device=device)
    alive_rows = valid_rows[cfg.batch_rows : cfg.batch_rows + cfg.alive_rows]
    alive_ids = torch.tensor(alive_rows, device=device)
    dec = Decomposer(runner, layers, cfg)
    try:
        if cfg.center and cfg.resume is None:
            dec.set_means(torch.tensor(rows[-4 * cfg.batch_rows :], device=device))
        train(dec, cfg, rows, eval_ids)
        stats = {
            "model": cfg.model_name,
            "revision": cfg.revision,
            "data": {
                "dataset": PILE[0],
                "revision": PILE[1],
                "excluded_scan_docs": excluded,
                "validation_docs": VALIDATION_DOCS,
            },  # fmt: skip
            "settings": {k: v for k, v in asdict(cfg).items() if k not in PATHS},
            "final": evaluate(dec, eval_ids),
        }
        export(dec, cfg, alive_ids, stats)
    finally:
        dec.close()
    click.echo(f"Wrote {cfg.out}")


if __name__ == "__main__":
    main()
