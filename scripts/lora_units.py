"""Rank-1 LoRA adapter directions of Qwen2.5-0.5B as registry units (#5166).

Ward et al. (``docs/literature/rank-1-lora-reasoning-signals.md``) train a rank-1 LoRA on a
reasoning model, and read each adapter's ``lora_A`` as a unit: one scalar per token, the
projection of the adapted matrix's input on ``lora_A``. This script trains such a LoRA on
the MLP projections of every layer of Qwen2.5-0.5B, on s1K-1.1 as they did. The base
weights stay frozen.

``train`` fits the adapters and saves them, with the loss before and after on held-out
trajectories. ``units`` writes each ``down_proj`` adapter as two superposition files, for
``+lora_A`` and ``−lora_A``, which ``open-interpretability audit --superposition`` reads.
A ``down_proj`` adapter reads the MLP's hidden activations, as a superposition does. It
also writes each adapter's loading on its layer's VERIFIED poles.

    uv run --with datasets==4.4.1 python scripts/lora_units.py train --device cuda
    uv run python scripts/lora_units.py units
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("papers/lora-units")
PROJS = ("gate_proj", "up_proj", "down_proj")
SEED = 5166
TRAIN, HELD_OUT = 950, 50  # trajectories of s1K-1.1 used to fit and to measure the loss
MAX_TOKENS = 1024  # each trajectory is cut to this many tokens, to fit one 12 GB GPU
ACCUM, LR, ALPHA = 8, 1e-3, 1.0  # trajectories per optimiser step, learning rate, LoRA scale
EPOCHS = 2  # passes over the training trajectories
# lora_A starts this small, so that the direction it ends at is learned, not its random start.
INIT = 0.01
UNITS = 20  # down_proj adapter directions to audit, drawn with SEED


class RankOne(torch.nn.Module):
    """A frozen linear layer plus a rank-1 LoRA: ``W x + α (lora_A · x) lora_B``.

    ``lora_A`` starts small and random, and ``lora_B`` at zero, so the adapted layer starts
    as the base. A small start lets the learned part of ``lora_A`` dominate its final value.
    """

    def __init__(self, base: torch.nn.Linear, generator: torch.Generator):
        super().__init__()
        self.base = base
        a = INIT * torch.randn(base.in_features, generator=generator) / base.in_features**0.5
        self.lora_A = torch.nn.Parameter(a.to(base.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, device=base.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The base output plus the adapter's rank-1 write.

        Args:
            x: The layer's input, ``[..., in_features]``.
        """
        scalar = x @ self.lora_A.to(x.dtype)
        return self.base(x) + ALPHA * scalar.unsqueeze(-1) * self.lora_B.to(x.dtype)


def attach(model, seed: int = SEED) -> dict[str, RankOne]:
    """Wrap every MLP projection of ``model`` in a :class:`RankOne`, in place.

    Args:
        model: A Qwen2 causal language model.
        seed: The seed of the ``lora_A`` initialisation.

    Returns:
        The adapters by name, ``layers.<i>.<projection>``.
    """
    generator = torch.Generator().manual_seed(seed)
    adapters = {}
    for i, layer in enumerate(model.model.layers):
        for proj in PROJS:
            wrapped = RankOne(getattr(layer.mlp, proj), generator)
            setattr(layer.mlp, proj, wrapped)
            adapters[f"layers.{i}.{proj}"] = wrapped
    return adapters


def texts(rows) -> list[str]:
    """Each s1K-1.1 row as one plain-text trajectory: question, thinking and answer.

    Args:
        rows: Rows with ``question``, ``deepseek_thinking_trajectory`` and
            ``deepseek_attempt``.

    Examples:
        >>> texts([{"question": "Q", "deepseek_thinking_trajectory": "T",
        ...         "deepseek_attempt": "A"}])
        ['Question: Q\\n\\nThinking: T\\n\\nAnswer: A']
    """
    return [f"Question: {r['question']}\n\nThinking: {r['deepseek_thinking_trajectory']}"
            f"\n\nAnswer: {r['deepseek_attempt']}" for r in rows]  # fmt: skip


def split(n: int, seed: int = SEED) -> tuple[list[int], list[int]]:
    """Seeded, disjoint indices of the train and held-out trajectories.

    Args:
        n: The number of rows.
        seed: The seed.

    Examples:
        >>> train, held = split(1000)
        >>> len(train), len(held), set(train) & set(held)
        (950, 50, set())
    """
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order[:TRAIN], order[TRAIN : TRAIN + HELD_OUT]


def mean_loss(model, batches: list[torch.Tensor]) -> float:
    """The mean next-token loss over ``batches``, with no gradient.

    Args:
        model: The model.
        batches: Token ids, one trajectory each.
    """
    with torch.no_grad():
        return sum(float(model(b, labels=b).loss) for b in batches) / len(batches)


def cos_to_init(adapters: dict, initial: dict) -> dict[str, float]:
    """The median cosine of each projection's ``lora_A`` with its starting value.

    A value near 1 means the direction is still mostly its random start.

    Args:
        adapters: The trained adapters, from :func:`attach`.
        initial: Each adapter's ``lora_A`` before training, by name.
    """
    out = {}
    for proj in PROJS:
        cos = [float(torch.nn.functional.cosine_similarity(
                   a.lora_A.detach().float(), initial[name].float(), dim=0))
               for name, a in adapters.items() if name.endswith(proj)]  # fmt: skip
        out[proj] = round(statistics.median(cos), 4)
    return out


@click.group()
def main() -> None:
    """Train the rank-1 LoRA, then write its units."""


@main.command()
@click.option("--device", default="cuda", show_default=True)
def train(device: str) -> None:  # pragma: no cover - trains the model
    """Fit the adapters on s1K-1.1 and save them, with the held-out loss before and after.

    Args:
        device: The device of the model.
    """
    from datasets import load_dataset
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_dataset("simplescaling/s1K-1.1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    fit, held = split(len(rows))
    corpus = texts(rows)

    def ids(i):
        return tokenizer(corpus[i], return_tensors="pt").input_ids[:, :MAX_TOKENS].to(device)

    fit_ids, held_ids = [ids(i) for i in fit], [ids(i) for i in held]
    before = mean_loss(model, held_ids)
    adapters = attach(model)
    params = [p for a in adapters.values() for p in (a.lora_A, a.lora_B)]
    for p in params:
        p.data = p.data.float()
    optimiser = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    initial = {name: a.lora_A.detach().clone() for name, a in adapters.items()}
    # EPOCHS passes over the training trajectories, ACCUM of them per optimiser step.
    order = random.Random(SEED)
    for _ in range(EPOCHS):
        for step, i in enumerate(order.sample(range(len(fit_ids)), len(fit_ids))):
            (model(fit_ids[i], labels=fit_ids[i]).loss / ACCUM).backward()
            if (step + 1) % ACCUM == 0:
                optimiser.step()
                optimiser.zero_grad()
    after = mean_loss(model, held_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    tensors = {f"{name}.{part}": getattr(a, part).detach().float().cpu().contiguous()
               for name, a in adapters.items() for part in ("lora_A", "lora_B")}  # fmt: skip
    save_file(tensors, str(OUT / "adapters.safetensors"))
    meta = {
        "table": "training",
        "model": MODEL,
        "data": "simplescaling/s1K-1.1",
        "train": TRAIN,
        "held_out": HELD_OUT,
        "max_tokens": MAX_TOKENS,
        "accum": ACCUM,
        "lr": LR,
        "alpha": ALPHA,
        "seed": SEED,
        "epochs": EPOCHS,
        "init": INIT,
        "steps": EPOCHS * (len(fit_ids) // ACCUM),
        "cos_to_init": cos_to_init(adapters, initial),
        "held_out_loss_before": round(before, 5),
        "held_out_loss_after": round(after, 5),
    }
    (OUT / "training.json").write_text(json.dumps(meta, indent=1) + "\n")
    click.echo(json.dumps(meta))


def superposition(layer: int, direction: torch.Tensor, sign: int) -> dict:
    """The ``--superposition`` file of one adapter pole: every nonzero entry, signed.

    Args:
        layer: The adapter's layer.
        direction: The adapter's ``lora_A`` over the MLP's hidden units.
        sign: 1 for the ``+lora_A`` pole, -1 for ``−lora_A``.

    Examples:
        >>> superposition(3, torch.tensor([0.5, 0.0, -2.0]), -1)["members"]
        [{'neuron_idx': 0, 'weight': -0.5}, {'neuron_idx': 2, 'weight': 2.0}]
    """
    members = [{"neuron_idx": i, "weight": round(sign * float(w), 8)}
               for i, w in enumerate(direction.tolist()) if w != 0]  # fmt: skip
    pole = "positive" if sign > 0 else "negative"
    return {"model": MODEL, "layer": layer, "members": members, "method": "rank-one-lora",
            "source": f"down_proj lora_A of the rank-1 LoRA of papers/lora-units/, "
                      f"layer {layer}, {pole} side (#5166)"}  # fmt: skip


def draw(layers: int, n: int = UNITS, seed: int = SEED) -> list[tuple[int, int]]:
    """``n`` adapter poles ``(layer, sign)`` drawn without replacement, in a seeded order.

    Args:
        layers: The model's number of layers.
        n: How many to draw.
        seed: The seed.

    Examples:
        >>> picks = draw(24)
        >>> len(picks), len(set(picks)), all(s in (1, -1) for _, s in picks)
        (20, 20, True)
    """
    poles = [(layer, sign) for layer in range(layers) for sign in (1, -1)]
    return random.Random(seed).sample(poles, n)


def loading(direction: torch.Tensor, verified: list[int]) -> dict:
    """How much of an adapter's ``|lora_A|`` sits on its layer's VERIFIED neurons.

    Each neuron's percentile is its rank by ``|lora_A|`` among the layer's neurons, from 0
    (smallest) to 1. Random neurons give a median of about 0.5.

    Args:
        direction: The adapter's ``lora_A`` over the layer's neurons.
        verified: The layer's neurons with a VERIFIED pole.

    Examples:
        >>> got = loading(torch.tensor([0.1, 3.0, 0.2, 2.0]), [1, 3])
        >>> got["median_percentile"], got["share_of_norm"]
        (0.8333, 0.9962)
    """
    mags = direction.abs()
    order = mags.argsort()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    pct = [float(rank[i]) / (len(order) - 1) for i in verified]
    share = float((mags[verified] ** 2).sum() / (mags**2).sum()) if verified else None
    return {"verified": len(verified),
            "median_percentile": round(statistics.median(pct), 4) if pct else None,
            "share_of_norm": round(share, 4) if share is not None else None,
            "share_expected": round(len(verified) / len(order), 4)}  # fmt: skip


@main.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
def units(wiki_dir: Path) -> None:  # pragma: no cover - reads the registry and the adapters
    """Write the superposition files of the drawn poles, and every adapter's loadings.

    Args:
        wiki_dir: The registry that holds the VERIFIED records.
    """
    from safetensors.torch import load_file

    from openinterpretability.interference.keep import verified_records

    tensors = load_file(str(OUT / "adapters.safetensors"))
    layers = 1 + max(int(k.split(".")[1]) for k in tensors)
    verified: dict[int, set[int]] = {}
    for r in verified_records(wiki_dir, MODEL):
        ident = r.submission.identifier
        verified.setdefault(ident.layer, set()).add(ident.neuron_idx)
    rows = []
    for layer in range(layers):
        a = tensors[f"layers.{layer}.down_proj.lora_A"]
        rows.append({"table": "loading", "layer": layer, "norm": round(float(a.norm()), 6),
                     **loading(a, sorted(verified.get(layer, ())))})  # fmt: skip
    (OUT / "units").mkdir(parents=True, exist_ok=True)
    for i, (layer, sign) in enumerate(draw(layers)):
        unit = superposition(layer, tensors[f"layers.{layer}.down_proj.lora_A"], sign)
        name = f"L{layer}-{'pos' if sign > 0 else 'neg'}.json"
        (OUT / "units" / name).write_text(json.dumps(unit) + "\n")
        rows.append({"table": "unit", "order": i, "layer": layer, "sign": sign, "file": name})
    (OUT / "results.units.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"{len(rows)} rows; units in {OUT / 'units'}")


if __name__ == "__main__":
    main()
