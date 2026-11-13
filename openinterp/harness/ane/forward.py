"""A model's forward pass to one layer's MLP activations, written so the ANE computes it well.

Core ML runs a model on Apple's Neural Engine (ANE) only in fp16, and it can't convert
the HF modelling code. So :class:`ToLayer` runs the forward pass again on the HF weights.
In fp32 it agrees with HF to a relative error of about 1e-6. Five changes make it accurate
on the ANE (docs/analyses/ane-feasibility.md):

- RMSNorm divides each row by its largest value before it squares the row. Qwen2.5's
  residual stream gets to about 6,600, and 6,600² is more than fp16 holds.
- Attention scales q before ``q @ k^T``. Qwen2.5's products before scaling get to 1e5.
- It squares with ``x * x``, never ``pow``. The ANE computes ``pow`` incorrectly when
  ``pow`` is in a chain of operations.
- SiLU and GELU use ``exp``. The ANE's ``silu``, ``sigmoid`` and ``tanh`` are coarse.
- Each projection is wrapped in :class:`Rescaled`. A matmul whose products are small
  loses precision on the ANE.

It supports the GPT-2 and Qwen2 architectures (:data:`ARCHITECTURES`).

Nothing in the converted weights depends on the input shape. The causal mask, Qwen's
rotary tables and GPT-2's positions are inputs (:func:`shape_inputs`). The scales of
:class:`Rescaled` come from one sample of :data:`CALIBRATION` tokens, whatever the shape.
So every shape of one chunk has the same weights, and the cache keeps them once.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

ARCHITECTURES = ("gpt2", "qwen2")  # config.model_type values that ToLayer runs
WEIGHT_RMS = 0.25  # what rescale_projections scales each weight matrix's RMS to
# The largest scaled output that rescale_projections permits, on its random-token sample.
# Real text can go much higher. On a 1,604-token document, Qwen2.5-1.5B's layer-1
# down_proj reached 4 times its peak on random tokens. A cap of 16,384 gave it a scale of 8.
# Then the ANE's fp16 sums went past fp16's range, and the batch was 23% off. At 2,048,
# the projections that need a scale keep it. They have small outputs, e.g. o_proj near 4.
OUTPUT_PEAK = 2048.0
CALIBRATION = (4, 512)  # rows and seq of the random tokens that set the scales
MASK = -3e4  # added to the scores of later positions; fp16 holds it


def shape_inputs(config, seq: int, start: int = 0) -> dict[str, torch.Tensor]:
    """The inputs, after the first, of a :class:`ToLayer` chunk at sequence length ``seq``.

    Every chunk has the causal ``mask``. Qwen2 also has the rotary ``cos`` and ``sin``,
    computed in fp32. GPT-2's first chunk has the ``positions`` of its position embeddings.
    """
    out = {"mask": torch.full((seq, seq), MASK).triu(1)}
    if config.model_type == "gpt2":
        if start == 0:
            out["positions"] = torch.arange(seq, dtype=torch.int32)
        return out
    head_dim = config.hidden_size // config.num_attention_heads
    theta = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq).float(), inv).repeat(1, 2)
    return {**out, "cos": angles.cos(), "sin": angles.sin()}


def supported(model: nn.Module) -> bool:
    """True if :class:`ToLayer` can run ``model``."""
    return getattr(getattr(model, "config", None), "model_type", None) in ARCHITECTURES


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's tanh GELU, the same as HF's ``gelu_new``, in a form the ANE computes well.

    It uses ``x * x * x`` for ``pow(x, 3)``, and ``1 / (1 + exp(-2z))`` for
    ``0.5 * (1 + tanh(z))``.
    """
    z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return x / (1.0 + torch.exp(-2.0 * z))


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up``, in a form the ANE computes well.

    If ``exp`` overflows, the result is 0, which is the correct limit.
    """
    return gate * up / (1.0 + torch.exp(-gate))


class Rescaled(nn.Module):
    """``inner(x) / scale``, with the weight and bias of ``inner`` multiplied by ``scale``.

    ``scale`` is a power of two, so the result is the same in fp32.
    """

    def __init__(self, inner: nn.Module, scale: float):
        super().__init__()
        self.inner, self.scale = inner, scale
        with torch.no_grad():
            inner.weight.mul_(scale)
            if inner.bias is not None:
                inner.bias.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The output of ``inner`` before the weights were scaled."""
        return self.inner(x) / self.scale


def calibration_ids(config) -> torch.Tensor:
    """The random token ids that set the scales: :data:`CALIBRATION`, within the context."""
    context = getattr(config, "n_positions", None) or config.max_position_embeddings
    rows, seq = CALIBRATION
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, config.vocab_size, (rows, min(seq, context)), generator=generator)


class ToLayer(nn.Module):
    """GPT-2 or Qwen2 from layer ``start`` to ``layer``, on ``rows × seq`` tokens.

    With ``start`` 0 it reads token ids ``[rows, seq]``, else the residual stream
    ``[rows, seq, d_model]``. It returns the MLP activations of ``layer``,
    ``[rows, seq, width]``, or with ``resid_out`` the residual stream after ``layer``. That
    is one chunk of a pass (``chunks.py``). The other inputs are :func:`shape_inputs`; a
    call without them uses those of ``seq``. ``sample`` calibrates the scales of
    :class:`Rescaled`: token ids, or for ``start`` > 0 the residual stream, of any shape.
    The default is :data:`CALIBRATION` random token ids. It changes the weights of
    ``model``, so give it a copy that nothing else uses.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        rows: int,
        seq: int,
        start: int = 0,
        resid_out: bool = False,
        sample: torch.Tensor | None = None,
    ):
        super().__init__()
        self.start, self.resid_out = start, resid_out
        cfg = model.config
        self.gpt2 = cfg.model_type == "gpt2"
        self.body = model.transformer if self.gpt2 else model.model
        self.layer, self.rows, self.seq = layer, rows, seq
        self.nh = cfg.n_head if self.gpt2 else cfg.num_attention_heads
        self.nkv = self.nh if self.gpt2 else cfg.num_key_value_heads
        self.hd = (cfg.n_embd if self.gpt2 else cfg.hidden_size) // self.nh
        self.config = cfg
        if not self.gpt2:
            self.eps_root = cfg.rms_norm_eps**0.5
        self.extra = shape_inputs(cfg, seq, start)
        if sample is None:
            sample = calibration_ids(cfg)
        with self.shaped(*sample.shape[:2]):
            rescale_projections(self, self.blocks(), sample)

    @contextmanager
    def shaped(self, rows: int, seq: int) -> Iterator[None]:
        """Run at ``rows × seq`` for a while, e.g. to calibrate on a sample of another shape."""
        saved = self.rows, self.seq, self.extra
        self.rows, self.seq = rows, seq
        self.extra = shape_inputs(self.config, seq, self.start)
        try:
            yield
        finally:
            self.rows, self.seq, self.extra = saved

    def names(self) -> list[str]:
        """The names of the inputs after the first, in order."""
        return list(self.extra)

    def blocks(self) -> nn.ModuleList:
        """The decoder layers."""
        return self.body.h if self.gpt2 else self.body.layers

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm, the same as HF's, with no square larger than the row size."""
        peak = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        s, e = x / peak, self.eps_root / peak
        return weight * (s * torch.rsqrt((s * s).mean(-1, keepdim=True) + e * e))

    def rope(self, x: torch.Tensor, extra: dict) -> torch.Tensor:
        """Rotary position embedding, in HF's half-split layout."""
        h = self.hd // 2
        return x * extra["cos"] + torch.cat([-x[..., h:], x[..., :h]], -1) * extra["sin"]

    def attend(self, q, k, v, extra: dict | None = None) -> torch.Tensor:
        """Causal attention over ``[rows, seq, heads × dim]`` q, k and v."""
        extra = self.extra if extra is None else extra
        q, k, v = (t.view(self.rows, self.seq, -1, self.hd).transpose(1, 2) for t in (q, k, v))
        if not self.gpt2:
            q, k = self.rope(q, extra), self.rope(k, extra)
            rep = self.nh // self.nkv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scores = (q * self.hd**-0.5) @ k.transpose(-1, -2) + extra["mask"]
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(self.rows, self.seq, -1)

    def forward(self, inp: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
        """The MLP activations of ``self.layer``, or the residual stream after it."""
        named = dict(zip(self.names(), extra, strict=True)) if extra else self.extra
        return self._gpt2(inp, named) if self.gpt2 else self._qwen(inp, named)

    def _gpt2(self, inp: torch.Tensor, extra: dict) -> torch.Tensor:
        b = self.body
        x = inp
        if self.start == 0:
            x = b.wte(inp) + b.wpe(extra["positions"])
        for i in range(self.start, self.layer + 1):
            blk = b.h[i]
            qkv = blk.attn.c_attn(blk.ln_1(x)).chunk(3, -1)
            x = x + blk.attn.c_proj(self.attend(*qkv, extra))
            hidden = gelu_new(blk.mlp.c_fc(blk.ln_2(x)))
            if i == self.layer and not self.resid_out:
                return hidden
            x = x + blk.mlp.c_proj(hidden)
        return x

    def _qwen(self, inp: torch.Tensor, extra: dict) -> torch.Tensor:
        x = self.body.embed_tokens(inp) if self.start == 0 else inp
        for i in range(self.start, self.layer + 1):
            blk = self.body.layers[i]
            at, mlp = blk.self_attn, blk.mlp
            h = self.rms(x, blk.input_layernorm.weight)
            x = x + at.o_proj(self.attend(at.q_proj(h), at.k_proj(h), at.v_proj(h), extra))
            h = self.rms(x, blk.post_attention_layernorm.weight)
            hidden = swiglu(mlp.gate_proj(h), mlp.up_proj(h))
            if i == self.layer and not self.resid_out:
                return hidden
            x = x + mlp.down_proj(hidden)
        return x


def rescale_projections(net: ToLayer, blocks: nn.ModuleList, sample: torch.Tensor) -> None:
    """Wrap each projection that ``net`` runs in :class:`Rescaled`.

    The scale makes the RMS of the weights about :data:`WEIGHT_RMS`. One fp32 pass on
    ``sample`` measures each output's peak. The scale stays small enough that the
    scaled peak is at most :data:`OUTPUT_PEAK`, a margin below fp16's largest value.
    """
    found = {p: m for p, m in blocks.named_modules() if type(m).__name__ in ("Linear", "Conv1D")}
    peaks: dict[str, float] = {}
    hooks = [
        m.register_forward_hook(
            lambda _m, _a, out, p=p: peaks.__setitem__(p, float(out.abs().max()))
        )
        for p, m in found.items()
    ]
    with torch.no_grad():
        net(sample)
    for hook in hooks:
        hook.remove()
    for path, module in found.items():
        if path not in peaks:  # after the layer that the pass returns
            continue
        rms = float(module.weight.detach().pow(2).mean().sqrt())
        scale = 2.0 ** round(math.log2(WEIGHT_RMS / rms))
        while scale > 1 and peaks[path] * scale > OUTPUT_PEAK:
            scale /= 2
        parent, attr = path.rsplit(".", 1)
        setattr(blocks.get_submodule(parent), attr, Rescaled(module, scale))
