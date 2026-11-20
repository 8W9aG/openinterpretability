"""Fluent dreaming: prompts optimised to drive a neuron pole as high as it goes (issue #3228).

This is EPO, from Thompson, Straznickas and Sklar (2024), "Fluent dreaming for language
models", arXiv:2402.01702, §2. A population of prompts of ``n`` tokens has one member for
each λ. Member i maximises

    L_λ(t) = f(t) − (λ / n) · Σ_i H(m(t_≤i), t_{i+1})

where f is the pole's strength and H is the model's cross-entropy on its own prompt. So a
larger λ asks for more fluent text. Each step backpropagates L to a one-hot encoding of
each prompt, and takes the ``k`` tokens with the largest gradient at each position. Each
member then has ``r`` children, each with one position, chosen at random, swapped for one
of its top-``k`` tokens. Each λ keeps the best prompt of the pooled population and
children, with replacement.

f here is the pole's largest strength over the prompt, with the first token left out:
many units take extreme values there (``docs/PROTOCOL.md`` §1). This version has no
restarts, and the defaults for ``k`` and the number of steps are lower than the paper's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from openinterpretability.harness.hooks import NeuronHookRunner, pole_strength

LENGTH = 12
LAMBDAS = tuple(math.exp(x) for x in torch.linspace(math.log(0.1), math.log(10), 8).tolist())
CHILDREN = 32
TOP_K = 128
STEPS = 150


@dataclass
class Dream:
    """One λ's final prompt: its tokens, text, pole strength and self-cross-entropy."""

    lam: float
    tokens: list[int]
    text: str
    strength: float
    cross_entropy: float


def _score(runner, layer, neuron, pole, embeds: torch.Tensor, tokens: torch.Tensor):
    """(f, mean cross-entropy) of each prompt, from its input embeddings ``[B, n, d]``."""
    captured = {}

    def grab(_module, args):
        captured["act"] = args[0][..., neuron]

    handle = runner.down_proj(layer).register_forward_pre_hook(grab)
    try:
        logits = runner.model(inputs_embeds=embeds).logits
    finally:
        handle.remove()
    strength = pole_strength(captured["act"], pole)[:, 1:].max(dim=1).values
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(tokens.shape[0], -1).mean(dim=1)  # fmt: skip
    return strength.float(), ce


def dream(
    runner: NeuronHookRunner,
    layer: int,
    neuron: int,
    pole: str,
    *,
    length: int = LENGTH,
    lambdas: Sequence[float] = LAMBDAS,
    children: int = CHILDREN,
    top_k: int = TOP_K,
    steps: int = STEPS,
    seed: int = 0,
) -> list[Dream]:
    """Run EPO for one pole and return each λ's final prompt."""
    gen = torch.Generator().manual_seed(seed)
    table = runner.model.get_input_embeddings().weight
    vocab, device = table.shape[0], table.device
    lams = torch.tensor(lambdas, device=device)
    pop = torch.randint(vocab, (len(lambdas), length), generator=gen).to(device)
    for _ in range(steps):
        onehot = torch.nn.functional.one_hot(pop, vocab).to(table.dtype).requires_grad_(True)
        f, ce = _score(runner, layer, neuron, pole, onehot @ table, pop)
        (f - lams * ce).sum().backward()
        top = onehot.grad.topk(top_k, dim=-1).indices  # [M, n, k]
        member = torch.arange(len(lambdas)).repeat_interleave(children)
        where = torch.randint(length, (len(member),), generator=gen)
        pick = torch.randint(top_k, (len(member),), generator=gen)
        kids = pop[member].clone()
        kids[torch.arange(len(member)), where.to(device)] = top[
            member.to(device), where.to(device), pick.to(device)
        ]
        pool = torch.cat([pop, kids])
        with torch.no_grad():
            f_pool, ce_pool = _score(runner, layer, neuron, pole, table[pool], pool)
        best = (f_pool[None, :] - lams[:, None] * ce_pool[None, :]).argmax(dim=1)
        pop = pool[best]
    with torch.no_grad():
        f, ce = _score(runner, layer, neuron, pole, table[pop], pop)
    return [
        Dream(
            round(float(lam), 4),
            pop[i].tolist(),
            runner.tokenizer.decode(pop[i].tolist()),
            round(float(f[i]), 4),
            round(float(ce[i]), 4),
        )  # fmt: skip
        for i, lam in enumerate(lambdas)
    ]
