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

Three goals replace f in L (#3779):

- ``max``: f, the paper's goal. It finds inputs that fire the pole.
- ``min``: −f. It starts from a prompt with some positions frozen. So it keeps an
  explanation's trigger tokens, and finds the context that silences the pole: a false
  positive.
- ``hit``: −|f − target|. It finds inputs at a set strength, such as the active line, to
  show how sharp an explanation's boundary is.
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
GOALS = ("max", "min", "hit")


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


def _aim(f: torch.Tensor, goal: str, target: float | None) -> torch.Tensor:
    """The goal's term of L (module docstring).

    Args:
        f: The pole strengths.
        goal: ``max``, ``min`` or ``hit``.
        target: The strength that ``hit`` aims at.

    Returns:
        The term, one value per strength.
    """
    if goal == "max":
        return f
    return -f if goal == "min" else -(f - target).abs()


def _start(members: int, length: int, vocab: int, gen, start, frozen) -> tuple:
    """The first population, and the positions that may change.

    Args:
        members: One per λ.
        length: Tokens per prompt, when ``start`` is not given.
        vocab: The vocabulary size.
        gen: The random generator.
        start: Token ids that every member starts from, or None for random tokens.
        frozen: Positions that never change.

    Returns:
        The population ``[members, n]`` on the CPU, and the free positions.

    Raises:
        ValueError: Every position is frozen.
    """
    if start is not None:
        pop = torch.tensor(list(start)).repeat(members, 1)
    else:
        pop = torch.randint(vocab, (members, length), generator=gen)
    free = torch.tensor([i for i in range(pop.shape[1]) if i not in set(frozen)])
    if not len(free):
        raise ValueError("every position is frozen")
    return pop, free


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
    goal: str = "max",
    target: float | None = None,
    start: Sequence[int] | None = None,
    frozen: Sequence[int] = (),
) -> list[Dream]:
    """Run EPO for one pole and return each λ's final prompt.

    Args:
        runner: The model, with hooks.
        layer: The pole's layer.
        neuron: The pole's neuron.
        pole: ``positive`` or ``negative``.
        length: Tokens per prompt, when ``start`` is not given.
        lambdas: The fluency weights, one population member each.
        children: Children per member per step.
        top_k: Candidate tokens per position.
        steps: Optimisation steps.
        seed: The random seed.
        goal: ``max``, ``min`` or ``hit`` (module docstring).
        target: The strength that ``hit`` aims at.
        start: Token ids that every member starts from, instead of random tokens.
        frozen: Positions of ``start`` that never change.

    Returns:
        Each λ's final prompt, with its strength and self-cross-entropy.

    Raises:
        ValueError: The goal is unknown, ``hit`` has no target, or no position is free.
    """
    if goal not in GOALS or (goal == "hit") != (target is not None):
        raise ValueError(f"goal must be one of {GOALS}, with a target for 'hit' only")
    gen = torch.Generator().manual_seed(seed)
    table = runner.model.get_input_embeddings().weight
    vocab, device = table.shape[0], table.device
    lams = torch.tensor(lambdas, device=device)
    pop, free = _start(len(lambdas), length, vocab, gen, start, frozen)
    pop, length = pop.to(device), pop.shape[1]

    for _ in range(steps):
        onehot = torch.nn.functional.one_hot(pop, vocab).to(table.dtype).requires_grad_(True)
        f, ce = _score(runner, layer, neuron, pole, onehot @ table, pop)
        (_aim(f, goal, target) - lams * ce).sum().backward()
        top = onehot.grad.topk(top_k, dim=-1).indices  # [M, n, k]
        member = torch.arange(len(lambdas)).repeat_interleave(children)
        where = free[torch.randint(len(free), (len(member),), generator=gen)]
        pick = torch.randint(top_k, (len(member),), generator=gen)
        kids = pop[member].clone()
        kids[torch.arange(len(member)), where.to(device)] = top[
            member.to(device), where.to(device), pick.to(device)
        ]
        pool = torch.cat([pop, kids])
        with torch.no_grad():
            f_pool, ce_pool = _score(runner, layer, neuron, pole, table[pool], pool)
        pop = pool[(_aim(f_pool, goal, target) - lams[:, None] * ce_pool).argmax(dim=1)]
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
