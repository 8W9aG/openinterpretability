"""Apply a patch chain to a model in memory, and remove it again (issue #5229).

Every operation is a forward hook, so the model's own weights are never written, and
removing the hooks restores the base model exactly. A weight edit ΔW on a layer adds
``x @ ΔWᵀ`` to that layer's output, which is what editing the weight would do.

``strength`` scales every operation's effect: 1 applies the patch, and 0 must give the base
model (:func:`zero_start_check`). A non-additive op mixes, ``x + strength·(edited − x)``.
A gated op is further multiplied by its gate's output at each position.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from openinterpretability.patches import tensors
from openinterpretability.patches.gates import Gates, inputs_of
from openinterpretability.patches.graph import Graph, hook_point, parameter_hook, parse
from openinterpretability.patches.schema import OP_TYPES, Op, PatchError

FLOAT32_NOISE = 1e-4  # largest |Δ logit| that the zero-start check calls equal (#4749)
ACTS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": nn.functional.gelu, "relu": torch.relu, "silu": nn.functional.silu,
}  # fmt: skip


class ZeroStartError(AssertionError):
    """A patch at zero strength changed the base model's logits."""


class _Context:
    """What the hooks of one applied chain share."""

    def __init__(self, model: nn.Module, gates: dict[str, dict], strength: float, root):
        self.graph = Graph(model)
        self.strength, self.root = strength, root
        self.gates = Gates(gates, self.load)
        self.handles: list = []

    def load(self, value) -> torch.Tensor:
        return tensors.load(value, self.root)

    def factor(self, op: Op, ref: str, like: torch.Tensor) -> torch.Tensor:
        """``strength`` times the op's own ``scale``, times its gate, shaped to ``like``."""
        f = self.strength * float(op.params.get("scale", 1.0))
        if op.gate is None:
            return torch.tensor(f, device=like.device)
        g = self.gates.output(_resolve(op.gate, ref)).to(like.device)
        return f * g.reshape(*g.shape, *([1] * (like.dim() - 2)))

    def at(self, point: str, fn) -> None:
        self.handles.append(hook_point(self.graph, point, fn))


def _resolve(name: str, ref: str) -> str:
    """A gate name as a full ref: a bare op_id belongs to the same patch as ``ref``."""
    return name if ":" in name else f"{ref.split(':')[0]}:{name}"


def _delta(ctx: _Context, op: Op, target: str) -> torch.Tensor:
    """The change ΔW (or Δb) that a weight edit makes to parameter ``target``."""
    p, kind = op.params, op.type
    if kind == "dense_add":
        return ctx.load(p["delta"])
    if kind == "lowrank_add":
        return ctx.load(p["B"]) @ ctx.load(p["A"])
    current = ctx.graph.weight(target)
    if kind == "sparse_add":
        delta = torch.zeros_like(current)
        index = [int(i) for i in p["index"]]
        values = ctx.load(p["values"])
        if p["axis"] == "row" or current.dim() == 1:
            delta[index] = values
        else:
            delta[:, index] = values
        return delta
    if kind == "replace":
        return ctx.load(p["value"]) - current
    return (float(p["factor"]) - 1.0) * current  # scale


def _weight(ctx: _Context, ref: str, op: Op) -> None:
    (target,) = op.targets
    delta = _delta(ctx, op, target)
    bias = parse(target).part.split(".")[-1].startswith("b_")

    def add(x: torch.Tensor) -> torch.Tensor:
        term = delta.to(x.device).expand(*x.shape[:-1], -1) if bias else x @ delta.to(x.device).t()
        return ctx.factor(op, ref, term) * term

    ctx.handles.extend(parameter_hook(ctx.graph, target, add))


def _act(ctx: _Context, op: Op, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
    name = op.params.get("act")
    if name:
        return ACTS[name]
    mlp = ctx.graph.block(layer, "mlp")
    return getattr(mlp, "act", None) or mlp.act_fn


def _add_neurons(ctx: _Context, ref: str, op: Op) -> None:
    """New MLP neurons, read from the MLP's input and added to its output."""
    (target,) = op.targets  # blocks.<i>.mlp.hook_out
    layer = parse(target).layer
    p = op.params
    w_in, w_out = ctx.load(p["W_in"]), ctx.load(p["W_out"])
    b_in = ctx.load(p["b_in"]) if "b_in" in p else torch.zeros(w_in.shape[0])
    w_gate = ctx.load(p["W_gate"]) if "W_gate" in p else None
    act, seen = _act(ctx, op, layer), []
    ctx.at(f"blocks.{layer}.mlp.hook_in", lambda x: seen.append(x) or None)

    def add(out: torch.Tensor) -> torch.Tensor:
        x = seen.pop().float()
        pre = x @ w_in.to(x.device).t() + b_in.to(x.device)
        h = act(x @ w_gate.to(x.device).t()) * pre if w_gate is not None else act(pre)
        term = h @ w_out.to(x.device).t()
        return out + (ctx.factor(op, ref, term) * term).to(out.dtype)

    ctx.at(target, add)


def _module(ctx: _Context, spec: dict, x: torch.Tensor) -> torch.Tensor:
    """Run a small block from the module registry (schema.MODULES) on ``x``."""
    if spec.get("ln", False):
        x = nn.functional.layer_norm(x, x.shape[-1:])

    def affine(x: torch.Tensor, w: str, b: str) -> torch.Tensor:
        return x @ ctx.load(spec[w]).to(x.device).t() + ctx.load(spec.get(b, 0.0)).to(x.device)

    if spec["kind"] == "linear":
        return affine(x, "W", "b")
    if spec["kind"] != "mlp":
        raise PatchError(f"unknown module kind {spec['kind']!r}")
    return affine(ACTS[spec.get("act", "gelu")](affine(x, "W1", "b1")), "W2", "b2")


def _residual(ctx: _Context, ref: str, op: Op, spec: dict) -> None:
    """A residual-additive block: it reads ``input`` and adds its output at the target."""
    (target,) = op.targets
    source = op.params.get("input", target)
    seen: list[torch.Tensor] = []
    if source != target:
        ctx.at(source, lambda x: seen.append(x) or None)

    def add(out: torch.Tensor) -> torch.Tensor:
        x = (seen.pop() if seen else out).float()
        term = _module(ctx, spec, x)
        return out + (ctx.factor(op, ref, term) * term).to(out.dtype)

    ctx.at(target, add)


def _structural(ctx: _Context, ref: str, op: Op) -> None:
    if op.type == "add_neurons":
        return _add_neurons(ctx, ref, op)
    if op.type == "add_layer":  # a residual MLP block after the target block
        spec = {"kind": "mlp", "ln": True, **{k: op.params[k] for k in op.params if k != "scale"}}
        return _residual(ctx, ref, op, spec)
    if op.type == "add_module":
        return _residual(ctx, ref, op, op.params["module"])
    return _prune(ctx, ref, op)


def _set_unit(ctx: _Context, ref: str, op: Op, point: str, index: slice | int, value) -> None:
    """Move the unit at ``index`` of ``point`` towards ``value``, by the op's factor."""

    def move(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        old = x[..., index].float()
        target = torch.as_tensor(value, device=x.device, dtype=torch.float32)
        f = ctx.factor(op, ref, old if old.dim() > 2 else old.unsqueeze(-1))
        f = f if old.dim() > 2 else f.squeeze(-1)
        x[..., index] = (old + f * (target - old)).to(x.dtype)
        return x

    ctx.at(point, move)


def _prune(ctx: _Context, ref: str, op: Op) -> None:
    """Zero or mean ablation of one neuron or one head (``value`` holds the mean)."""
    (target,) = op.targets
    layer, p = parse(target).layer, op.params
    value = float(p.get("value", 0.0)) if p.get("mode", "zero") == "mean" else 0.0
    if p["unit"] == "neuron":
        return _set_unit(ctx, ref, op, f"blocks.{layer}.mlp.hook_post", int(p["index"]), value)
    width = ctx.graph.block(layer, "W_O").weight.shape[-1] // ctx.graph.heads
    head = int(p["index"])
    span = slice(head * width, (head + 1) * width)
    return _set_unit(ctx, ref, op, f"blocks.{layer}.attn.hook_z", span, value)


def _activation(ctx: _Context, ref: str, op: Op) -> None:
    (target,) = op.targets
    if op.type == "clamp":
        return _set_unit(ctx, ref, op, target, int(op.params["index"]), op.params["value"])
    vector = ctx.load(op.params["vector"])  # steer

    def add(x: torch.Tensor) -> torch.Tensor:
        term = vector.to(x.device).expand(*x.shape[:-1], -1)
        return x + (ctx.factor(op, ref, term) * term).to(x.dtype)

    return ctx.at(target, add)


INSTALL = {"weight": _weight, "structural": _structural, "activation": _activation}


def _recorders(ctx: _Context, gates: dict[str, dict]) -> None:
    """Record every hook point that a gate reads, once per forward pass."""
    points = sorted({p for params in gates.values() for p in inputs_of(params["function"])})
    for point in points:
        ctx.at(point, lambda x, point=point: ctx.gates.values.__setitem__(point, x) or None)
    ctx.handles.append(ctx.graph.model.register_forward_pre_hook(lambda *_: ctx.gates.reset()))


@contextmanager
def applied(model: nn.Module, ops: Sequence[tuple[str, Op]], strength: float = 1.0,
            root: Path | None = None, observe: list | None = None):  # fmt: skip
    """Apply ``ops`` (from :func:`~.history.replay`) to ``model`` for the ``with`` block.

    With ``observe``, each forward pass appends the outputs of the gates it used, by ref.
    """
    gates = {ref: op.params for ref, op in ops if op.type == "gate"}
    for op in (op for _, op in ops if not OP_TYPES[op.type].built):
        raise PatchError(f"operation type {op.type!r} is in the registry but not built yet")
    ctx = _Context(model, gates, strength, root)
    eager = any(p["function"]["kind"] == "attention_score" for p in gates.values())
    old = model.config._attn_implementation
    try:
        if eager:
            model.set_attn_implementation("eager")
        _recorders(ctx, gates)
        if observe is not None:

            def record(*_) -> None:
                observe.append({ref: v.detach().cpu() for ref, v in ctx.gates.cache.items()})

            ctx.handles.append(model.register_forward_hook(record))
        for ref, op in ops:
            family = OP_TYPES[op.type].family
            if family in INSTALL:
                INSTALL[family](ctx, ref, op)
        yield model
    finally:
        for handle in ctx.handles:
            handle.remove()
        if eager:
            model.set_attn_implementation(old)


@torch.no_grad()
def logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """The model's logits on ``input_ids``, as float32."""
    return model(input_ids=input_ids).logits.float()


@torch.no_grad()
def zero_start_check(model: nn.Module, ops: Sequence[tuple[str, Op]], input_ids: torch.Tensor,
                     strength: float = 0.0, root: Path | None = None) -> float:  # fmt: skip
    """Raise :class:`ZeroStartError` unless the patch at ``strength`` keeps the base logits.

    The default, 0, checks the patch with every output scale at zero. Strength 1 checks ops
    that are the identity as they start (``identity_at_init``). Return the largest |Δ logit|.
    """
    base = logits(model, input_ids)
    with applied(model, ops, strength, root):
        patched = logits(model, input_ids)
    gap = float((patched - base).abs().max())
    if gap > FLOAT32_NOISE:
        raise ZeroStartError(f"patch at strength {strength} moves a logit by {gap:.3g}")
    return gap
