"""Arm F of the copy-family fix: a closed-form edit at one MLP (#7006, #5220 §3).

MEMIT's least squares fits one change of the layer's output weight for every train
prompt. The AlphaEdit variant projects the change off the retain prompts' keys, so those
keys give the same output as before. ``scripts/fix_arms.py`` writes the fit as a patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_family import Family, Fit, Item, Layout, encode, token  # noqa: E402

V_STEPS, V_LR, V_NORM = 40, 0.5, 4.0  # the fit of each prompt's target value
MOMENT = 1e4  # the weight of the corpus second moment in the solve, our setting
NULL_TOL = 1e-2  # a retain key direction is kept unchanged above this share of the largest


def keys_at(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    """The inputs of ``layer``'s MLP output weight at each prompt's last position, ``[hidden, n]``.

    Args:
        model: The model.
        tokenizer: Its tokenizer.
        prompts: The prompts.
        layer: The layer.
    """
    from openinterpretability.patches.graph import Graph, hook_point

    seen: list[torch.Tensor] = []
    handle = hook_point(Graph(model), f"blocks.{layer}.mlp.hook_post",
                        lambda x: seen.append(x.detach().float()) or None)  # fmt: skip
    try:
        device = next(model.parameters()).device
        ids, mask, last = encode(tokenizer, prompts, device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    finally:
        handle.remove()
    return seen[0][torch.arange(len(prompts), device=device), last].t()


def second_moment(model, batches: list[torch.Tensor], layer: int) -> torch.Tensor:
    """The mean of ``k kᵀ`` over every position of ``batches``, for ``layer``'s MLP keys.

    Args:
        model: The model.
        batches: Corpus windows, as batches of ids.
        layer: The layer.
    """
    from openinterpretability.patches.graph import Graph, hook_point

    total, count = None, 0

    def add(x):
        nonlocal total, count
        flat = x.detach().float().reshape(-1, x.shape[-1])
        total = flat.t() @ flat if total is None else total + flat.t() @ flat
        count += flat.shape[0]

    handle = hook_point(Graph(model), f"blocks.{layer}.mlp.hook_post", add)
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in batches:
                model(input_ids=batch.to(device))
    finally:
        handle.remove()
    return total / count


def target_deltas(model, tokenizer, items: list[Item], layer: int,
                  steps: int = V_STEPS) -> torch.Tensor:  # fmt: skip
    """The change of ``layer``'s MLP output that makes each prompt's right unit likely.

    Each change acts at the prompt's last position, and its norm is at most :data:`V_NORM`.
    The result is ``[d_model, n]``.

    Args:
        model: The model.
        tokenizer: Its tokenizer.
        items: The train prompts.
        layer: The layer.
        steps: Adam steps of the fit.
    """
    from openinterpretability.patches.graph import Graph, hook_point

    device = next(model.parameters()).device
    ids, mask, last = encode(tokenizer, [i.prompt for i in items], device)
    rows = torch.arange(len(items), device=device)
    right = torch.tensor([token(tokenizer, i.right) for i in items], device=device)
    delta = torch.zeros(len(items), model.config.hidden_size, device=device, requires_grad=True)

    def add(out):
        out = out.clone()
        out[rows, last] = out[rows, last] + delta.to(out.dtype)
        return out

    opt = torch.optim.Adam([delta], lr=V_LR)
    handle = hook_point(Graph(model), f"blocks.{layer}.mlp.hook_out", add)
    try:
        for _ in range(steps):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()[rows, last]
            loss = -torch.log_softmax(logits, -1)[rows, right].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                norms = delta.norm(dim=-1, keepdim=True).clamp(min=V_NORM)
                delta.mul_(V_NORM / norms)
    finally:
        handle.remove()
    return delta.detach().t()


def null_projector(keys: torch.Tensor, tol: float = NULL_TOL) -> torch.Tensor:
    """The projector onto the directions that ``keys`` (``[hidden, m]``) hardly use.

    Examples:
        >>> k = torch.tensor([[1.0], [0.0]])
        >>> null_projector(k)
        tensor([[0., 0.],
                [0., 1.]])

    Args:
        keys: The keys to keep unchanged, one per column.
        tol: A direction whose singular value is above ``tol`` times the largest is removed.
    """
    u, s, _ = torch.linalg.svd(keys, full_matrices=False)
    used = u[:, s > tol * s.max()]
    return torch.eye(keys.shape[0], device=keys.device) - used @ used.t()


def closed_form(keys: torch.Tensor, residuals: torch.Tensor, moment: torch.Tensor,
                projector: torch.Tensor | None = None,
                weight: float | None = None) -> torch.Tensor:  # fmt: skip
    """MEMIT's least squares at one layer: ``ΔW = R Kᵀ (K Kᵀ + w C)⁻¹``, ``[d_model, hidden]``.

    With ``projector`` P, AlphaEdit's form: ``ΔW = R Kᵀ P (P K Kᵀ P + w P C P + I)⁻¹``.
    Then ``ΔW k = 0`` for each key k that P removes.

    Args:
        keys: K, the train keys, ``[hidden, n]``.
        residuals: R, the change of output that each key must give, ``[d_model, n]``.
        moment: C, the corpus second moment of the keys, ``[hidden, hidden]``.
        projector: P, or None for MEMIT's form.
        weight: w, the weight of C. None reads :data:`MOMENT` at the call.

    Examples:
        >>> k = torch.eye(2)[:, :1]
        >>> r = torch.tensor([[2.0]])
        >>> closed_form(k, r, torch.eye(2), weight=1.0)  # 2·kᵀ(k kᵀ + I)⁻¹
        tensor([[1., 0.]])
    """
    weight = MOMENT if weight is None else weight
    k, r, c = keys.double(), residuals.double(), moment.double()  # the solve needs float64
    if projector is None:
        inner, left = k @ k.t() + weight * c, r @ k.t()
    else:
        p = projector.double()
        inner = p @ k @ k.t() @ p + weight * p @ c @ p + torch.eye(k.shape[0], device=k.device,
                                                                     dtype=k.dtype)  # fmt: skip
        left = r @ k.t() @ p
    # inner is symmetric, so left @ inner⁻¹ is the transpose of inner⁻¹ @ leftᵀ.
    return torch.linalg.solve(inner, left.t()).t().to(keys.dtype)


def low_rank(delta: torch.Tensor, rank: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The best rank-``rank`` factors ``(A, B)`` of ``delta``, with ``B @ A ≈ delta``.

    Args:
        delta: The change, ``[out, in]``.
        rank: The rank, or None for every direction above rounding: the edit as solved.

    Examples:
        >>> d = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        >>> a, b = low_rank(d, 1)
        >>> (b @ a).round()
        tensor([[3., 0.],
                [0., 0.]])
    """
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    r = int((s > s.max() * 1e-6).sum()) if rank is None else min(rank, s.shape[0])
    return vh[:r], u[:, :r] * s[:r]


def fit_closed_form(runner, family: Family, layout: Layout, batches: list[torch.Tensor],
                    rank: int | None, variant: str = "memit") -> Fit:  # fmt: skip
    """Fit arm F at B's middle layer, as MEMIT or as AlphaEdit, cut to rank ``rank``.

    Args:
        runner: The base model's runner.
        family: The family.
        layout: Where the arms act.
        batches: Corpus windows, for the second moment.
        rank: The rank of the change, or None for the edit as solved.
        variant: ``memit``, or ``alphaedit`` to keep the retain prompts' keys unchanged.
    """
    model, tokenizer = runner.model, runner.tokenizer
    layer = layout.fact_layers[len(layout.fact_layers) // 2]
    keys = keys_at(model, tokenizer, [i.prompt for i in family.train], layer)
    residuals = target_deltas(model, tokenizer, family.train, layer)
    moment = second_moment(model, batches, layer)
    projector = None
    if variant == "alphaedit" and family.retain:
        kept = keys_at(model, tokenizer, [i.prompt for i in family.retain], layer)
        projector = null_projector(kept)
    a, b = low_rank(closed_form(keys, residuals, moment, projector), rank)
    return Fit("F", {"variant": variant, "rank": rank if rank is not None else "full"},
               [(f"blocks.{layer}.mlp.W_out", a, b)])  # fmt: skip
