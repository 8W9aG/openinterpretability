"""Gates: the parts of a patch that decide where its operations act (issue #5229).

A gate is a ``gate`` op. Other ops name it by op_id, and their effect is multiplied by its
output, a value in [0, 1] for each position. ``params``:

- ``function``: a dict whose ``kind`` is one of :data:`~.schema.GATE_FUNCTIONS`.
  - ``linear_probe``: ``input`` (a hook point), ``w`` and ``b``. The score is ``x·w + b``.
  - ``threshold``: ``input``, and ``index`` (a unit) or ``direction``. The score is the
    unit's value, or ``x·direction``.
  - ``token_match``: ``tokens``. The score is 1 where the input token is one of them.
  - ``attention_score``: ``heads`` (a list of ``[layer, head]``) and ``key_tokens``. The
    score is the heads' mean attention from each position to the keys with those tokens.
  - ``mlp``: ``input``, ``W1``, ``b1``, ``W2`` and ``b2``, a small classifier.
  - ``and``, ``or``, ``not``, ``min`` and ``product``: ``gates``, other gates by op_id.
- ``output``: ``hard`` (score above ``threshold``) or ``soft`` (a sigmoid of the score less
  ``threshold``, over ``temperature``). A composite combines its gates' outputs as they are.
- ``positions``: ``all`` (the default), ``last``, or a list of positions. Outside them the
  gate is shut.
- ``force``: ``open`` or ``shut`` fixes the output at every position, for checks. A forced
  gate reads no input, so it also works outside a forward pass.
- ``meaning`` and ``metrics``: the gate's stated meaning and its measures, as data only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from openinterpretability.patches.schema import PatchError

Load = Callable[[Any], torch.Tensor]  # reads a tensor parameter (tensors.load)
COMPOSITES = ("and", "or", "not", "min", "product")


def inputs_of(function: dict) -> list[str]:
    """The hook points that a gate function reads, so they can be recorded."""
    kind = function.get("kind")
    if kind in ("linear_probe", "threshold", "mlp"):
        return [function["input"]]
    if kind == "token_match":
        return ["hook_tokens"]
    if kind == "attention_score":
        layers = sorted({int(layer) for layer, _ in function["heads"]})
        return ["hook_tokens", *(f"blocks.{layer}.attn.hook_pattern" for layer in layers)]
    if kind in COMPOSITES:
        return []
    raise PatchError(f"unknown gate function {kind!r}")


def _score(function: dict, values: dict[str, torch.Tensor], load: Load) -> torch.Tensor:
    """The raw score of a non-composite gate function, ``[batch, seq]``."""
    kind = function["kind"]
    if kind == "token_match":
        tokens = values["hook_tokens"]
        wanted = torch.as_tensor(function["tokens"], device=tokens.device)
        return torch.isin(tokens, wanted).float()
    if kind == "attention_score":
        return _attention(function, values)
    x = _recorded(values, function["input"]).float()
    if kind == "linear_probe":
        return x @ load(function["w"]).to(x.device) + float(function.get("b", 0.0))
    if kind == "threshold":
        if "index" in function:
            return x[..., int(function["index"])]
        return x @ load(function["direction"]).to(x.device)
    w1, b1 = load(function["W1"]).to(x.device), load(function["b1"]).to(x.device)
    w2, b2 = load(function["W2"]).to(x.device), load(function["b2"]).to(x.device)
    out = torch.relu(x @ w1.t() + b1) @ w2.t() + b2
    return out.squeeze(-1)


def _recorded(values: dict[str, torch.Tensor], point: str) -> torch.Tensor:
    if point not in values:
        raise PatchError(f"a gate reads {point}, which the forward pass hasn't reached yet")
    return values[point]


def _attention(function: dict, values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean attention, over the listed heads, from each position to the matching keys."""
    tokens = values["hook_tokens"]
    keys = torch.isin(tokens, torch.as_tensor(function["key_tokens"], device=tokens.device))
    scores = []
    for layer, head in function["heads"]:
        pattern = _recorded(values, f"blocks.{int(layer)}.attn.hook_pattern")
        if pattern is None:
            raise PatchError("attention weights are missing: the model must run eager attention")
        weights = pattern[:, int(head)].float()  # [batch, query, key]
        scores.append((weights * keys[:, None, :].float()).sum(-1))
    return torch.stack(scores).mean(0)


def _mask(params: dict, shape: torch.Size, device) -> torch.Tensor:
    """1 at the positions the gate may open, 0 elsewhere."""
    where = params.get("positions", "all")
    mask = torch.zeros(shape, device=device)
    if where == "all":
        mask += 1
    elif where == "last":
        mask[:, -1] = 1
    else:
        mask[:, [int(p) for p in where]] = 1
    return mask


class Gates:
    """The gates of one applied patch chain, evaluated once per forward pass."""

    def __init__(self, gates: dict[str, dict], load: Load):
        self.gates, self.load = gates, load
        self.values: dict[str, torch.Tensor] = {}
        self.cache: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        """Forget the last forward pass."""
        self.values.clear()
        self.cache.clear()

    def output(self, ref: str) -> torch.Tensor:
        """The gate's output for this forward pass, ``[batch, seq]`` in [0, 1]."""
        if ref not in self.cache:
            self.cache[ref] = self._compute(ref)
        return self.cache[ref]

    def _compute(self, ref: str) -> torch.Tensor:
        if ref not in self.gates:
            raise PatchError(f"no gate {ref}")
        params = self.gates[ref]
        force = params.get("force")
        if force in ("open", "shut"):  # fixed everywhere, so it reads no input
            return torch.tensor(1.0 if force == "open" else 0.0)
        function = params["function"]
        kind = function["kind"]
        if kind in COMPOSITES:
            value = _combine(kind, [self.output(child) for child in function["gates"]])
        else:
            score = _score(function, self.values, self.load)
            threshold = float(params.get("threshold", 0.0))
            if params.get("output", "hard") == "soft":
                value = torch.sigmoid((score - threshold) / float(params.get("temperature", 1.0)))
            else:
                value = (score > threshold).float()
        return value * _mask(params, value.shape, value.device)


def _combine(kind: str, parts: list[torch.Tensor]) -> torch.Tensor:
    """A composite gate's output from its gates' outputs."""
    if kind == "not":
        if len(parts) != 1:
            raise PatchError("a 'not' gate takes exactly one gate")
        return 1 - parts[0]
    stacked = torch.stack(parts)
    if kind in ("and", "min"):
        return stacked.min(0).values
    if kind == "or":
        return stacked.max(0).values
    return stacked.prod(0)
