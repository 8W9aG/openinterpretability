"""Tests of whether a fix removed the copy failure or only hid it, and the placebo arm (#7151).

The maintainer added these to the fitting harness of #7006 on 2026-11-27. They come from two
literature reviews: ``docs/literature/neuron-basis-applications.md`` and
``docs/literature/harness-evolution-ceiling.md``.

- **Cloze.** Each held-out wrong-copy prompt is read with two answers: the right unit and
  the copied unit, with no letters. The family's units are one token each. So this compares
  the two tokens at the last position. A fix hid the failure when the copied unit still
  ranks first.
- **Direction removal.** The direction is the fix's mean change to the residual stream. It
  is read after the last changed layer, over the train prompts. Projecting it out on the
  held-out prompts, with the fix applied, tests the fix. A fix that only hid the failure then
  loses its effect, and the copied unit comes back.
- **Placebo.** Arm P is arm B trained on the same prompts. Each prompt has a random unit word
  in place of its right unit. The answer stays a unit word, so the format doesn't break. Arm D
  asks whether *where* a fix goes matters, and P asks whether *what* it learns matters.
"""

from __future__ import annotations

import random
import re
from contextlib import contextmanager

import torch
from fix_family import Family, Item, encode, last_logits, measures, token

PLACEBO_SEED = 7151  # the draw of the placebo's units
_LAYER = re.compile(r"^blocks\.(\d+)\.")


def last_layer(targets: list[str]) -> int:
    """The highest block that a fit changes, from its ops' targets.

    Args:
        targets: Parameter names such as ``blocks.3.mlp.W_out``.

    Examples:
        >>> last_layer(["blocks.3.mlp.W_out", "blocks.5.mlp.W_out", "blocks.4.attn.W_O"])
        5
    """
    return max(int(_LAYER.match(t).group(1)) for t in targets)


@torch.no_grad()
def cloze_old_first(model, tokenizer, items: list[Item], prefix: str = "") -> float | None:
    """The share of ``items`` whose copied unit has a higher probability than the right unit.

    Args:
        model: The model, with any change hooked in.
        tokenizer: Its tokenizer.
        items: The held-out wrong-copy prompts.
        prefix: Text put before each prompt, for arm E.

    Returns:
        The share, rounded to 4 places, or None with no items.
    """
    if not items:
        return None
    logits = last_logits(model, tokenizer, [prefix + i.prompt for i in items])
    rows = torch.arange(len(items), device=logits.device)
    right = torch.tensor([token(tokenizer, i.right) for i in items], device=logits.device)
    copied = torch.tensor([token(tokenizer, i.copied) for i in items], device=logits.device)
    return round(float((logits[rows, copied] > logits[rows, right]).float().mean()), 4)


@torch.no_grad()
def resid_mean(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    """The mean residual stream after block ``layer``, over every real position of ``prompts``.

    Args:
        model: The model, with any change hooked in.
        tokenizer: Its tokenizer.
        prompts: The prompts. Padding positions are left out.
        layer: The block.
    """
    from openinterpretability.patches.graph import Graph, hook_point

    seen: list[torch.Tensor] = []
    handle = hook_point(Graph(model), f"blocks.{layer}.hook_resid_post",
                        lambda x: seen.append(x.detach()))  # fmt: skip
    device = next(model.parameters()).device
    ids, mask, _ = encode(tokenizer, prompts, device)
    try:
        model(input_ids=ids, attention_mask=mask)
    finally:
        handle.remove()
    return seen[0][mask.bool()].float().mean(0)


@contextmanager
def projected_out(model, layer: int, direction: torch.Tensor):
    """Project ``direction`` out of the residual stream after block ``layer``, at every position.

    Args:
        model: The model.
        layer: The block.
        direction: The direction, of any norm. A zero direction changes nothing.
    """
    from openinterpretability.patches.graph import Graph, hook_point

    norm = direction.norm()
    unit = direction / norm if norm > 0 else direction

    def remove(x):
        """``x`` less its component along the direction.

        Args:
            x: The residual stream, ``[batch, position, width]``.
        """
        u = unit.to(x.device, x.dtype)
        return x - (x @ u).unsqueeze(-1) * u

    handle = hook_point(Graph(model), f"blocks.{layer}.hook_resid_post", remove)
    try:
        yield
    finally:
        handle.remove()


def removal_tests(model, tokenizer, family: Family, ops, targets: list[str]) -> dict:
    """Both removal tests of one fit, on the family's held-out wrong copies.

    Args:
        model: The base model.
        tokenizer: Its tokenizer.
        family: The family. Its train prompts give the direction.
        ops: The fit's replayed patch ops, which :func:`~patches.apply.applied` applies.
        targets: The fit's parameter names, which give the layer of the direction.

    Returns:
        ``cloze_old_first`` with the fix, then ``direction``: the layer, the direction's norm,
        and the held-out measures and cloze share with the direction projected out.
    """
    from openinterpretability.patches.apply import applied

    layer = last_layer(targets)
    prompts = [i.prompt for i in family.train]
    base = resid_mean(model, tokenizer, prompts, layer)
    with applied(model, ops):
        direction = resid_mean(model, tokenizer, prompts, layer) - base
        cloze = cloze_old_first(model, tokenizer, family.heldout)
        with projected_out(model, layer, direction):
            removed = measures(model, tokenizer, family.heldout)
            removed["cloze_old_first"] = cloze_old_first(model, tokenizer, family.heldout)
    return {"cloze_old_first": cloze,
            "direction": {"layer": layer, "norm": round(float(direction.norm()), 4),
                          **removed}}  # fmt: skip


def base_row(model, tokenizer, family: Family) -> dict:
    """The base model's row: its measures on each split, and its held-out cloze share.

    Args:
        model: The base model.
        tokenizer: Its tokenizer.
        family: The family.
    """
    from fix_family import evaluate

    cloze = cloze_old_first(model, tokenizer, family.heldout)
    return {"table": "base", **evaluate(model, tokenizer, family),
            "removal": {"cloze_old_first": cloze}}  # fmt: skip


def placebo(family: Family, seed: int = PLACEBO_SEED) -> Family:
    """The family with each train prompt's right unit swapped for a random unit word.

    The units come from every right unit in the family. A prompt never gets its own right
    unit or the unit it copies. The held-out and retain prompts are unchanged, since the
    placebo is fitted on the train prompts and measured on the real family.

    Args:
        family: The family.
        seed: The seed of the draw.

    Raises:
        ValueError: The family has no unit to draw for some prompt.

    Examples:
        >>> f = Family([Item("a", " x", " y")], [], [Item("b", " q", " q")])
        >>> [i.right for i in placebo(f).train]
        [' q']
    """
    units = sorted({i.right for split in (family.train, family.heldout, family.retain,
                                          family.right_heldout) for i in split})  # fmt: skip
    rng, train = random.Random(seed), []
    for item in family.train:
        pool = [u for u in units if u not in (item.right, item.copied)]
        if not pool:
            raise ValueError(f"no placebo unit for {item.prompt!r}")
        train.append(Item(item.prompt, rng.choice(pool), item.copied))
    return Family(train, family.heldout, family.retain, family.right_heldout)
