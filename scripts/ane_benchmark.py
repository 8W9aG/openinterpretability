# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.7.0", "transformers==4.46.3", "coremltools==9.0", "numpy"]
# ///
"""Time a scan-sized forward pass on the CPU, on MPS and on Apple's Neural Engine (ANE).

A scan reads one layer's MLP activations for batches of 2,048 tokens (``BATCH_TOKENS``).
This script times that work for one model and layer, in these ways:

- PyTorch on the CPU (fp32), and on MPS (fp32 and fp16);
- Core ML (fp16) with each compute-unit setting: CPU only, CPU and GPU, CPU and ANE, all.

It also compares each result with the fp32 CPU activations, and it records on which
device Core ML puts each operation when the ANE is permitted. Results go to
docs/analyses/ane-feasibility.md.

The ANE is available only through Core ML, and Core ML cannot convert the HF modelling
code in transformers 5. Thus the model is run by :class:`ToLayer`, a forward pass
written again from the HF weights. The script makes sure that it agrees with HF in fp32.
The script has its own dependencies, so run it with uv, not in the project environment:

    HF_HUB_OFFLINE=1 uv run scripts/ane_benchmark.py openai-community/gpt2 5
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

BATCH, SEQ = 8, 256  # 2,048 tokens, as the scan's BATCH_TOKENS
REPS = 5
UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's tanh GELU, the same as HF's ``gelu_new``, with ``x * x * x`` for ``pow(x, 3)``.

    The ANE computes ``pow`` incorrectly (docs/analyses/ane-feasibility.md).
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class ToLayer(torch.nn.Module):
    """GPT-2 or Qwen2 up to ``layer``. It returns the MLP activations of that layer.

    It squares with ``x * x``, never ``pow``, because the ANE computes ``pow`` incorrectly.
    Two steps keep fp16 finite on Qwen2.5, where the residual stream gets to about 6,600.
    RMSNorm divides each row by its largest value before it squares the row. Attention
    scales q before ``q @ k^T``, because the products before scaling get to about 1e5.
    """

    def __init__(self, model: torch.nn.Module, layer: int):
        super().__init__()
        cfg = model.config
        self.gpt2 = hasattr(model, "transformer")
        self.body = model.transformer if self.gpt2 else model.model
        self.layer = layer
        self.nh = cfg.n_head if self.gpt2 else cfg.num_attention_heads
        self.nkv = self.nh if self.gpt2 else cfg.num_key_value_heads
        self.hd = (cfg.n_embd if self.gpt2 else cfg.hidden_size) // self.nh
        self.register_buffer("causal", torch.full((SEQ, SEQ), -3e4).triu(1))
        if not self.gpt2:
            self.eps_root = cfg.rms_norm_eps**0.5
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
            angles = torch.outer(torch.arange(SEQ).float(), inv).repeat(1, 2)
            self.register_buffer("cos", angles.cos())
            self.register_buffer("sin", angles.sin())

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm, the same as HF's, with no square larger than the row size."""
        peak = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        s, e = x / peak, self.eps_root / peak
        return weight * (s * torch.rsqrt((s * s).mean(-1, keepdim=True) + e * e))

    def rope(self, x: torch.Tensor) -> torch.Tensor:
        """Rotary position embedding, in HF's half-split layout."""
        h = self.hd // 2
        return x * self.cos + torch.cat([-x[..., h:], x[..., :h]], -1) * self.sin

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Causal attention over ``[batch, seq, heads × dim]`` q, k and v."""
        q, k, v = (t.view(BATCH, SEQ, -1, self.hd).transpose(1, 2) for t in (q, k, v))
        if not self.gpt2:
            q, k = self.rope(q), self.rope(k)
            rep = self.nh // self.nkv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scores = (q * self.hd**-0.5) @ k.transpose(-1, -2) + self.causal
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(BATCH, SEQ, -1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """The MLP activations of ``self.layer``, ``[batch, seq, width]``."""
        return self._gpt2(ids) if self.gpt2 else self._qwen(ids)

    def _gpt2(self, ids: torch.Tensor) -> torch.Tensor:
        b = self.body
        x = b.wte(ids) + b.wpe(torch.arange(SEQ, device=ids.device))
        for i, blk in enumerate(b.h[: self.layer + 1]):
            x = x + blk.attn.c_proj(self.attend(*blk.attn.c_attn(blk.ln_1(x)).chunk(3, -1)))
            hidden = gelu_new(blk.mlp.c_fc(blk.ln_2(x)))
            if i == self.layer:
                return hidden
            x = x + blk.mlp.c_proj(hidden)
        raise IndexError(self.layer)

    def _qwen(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.body.embed_tokens(ids)
        for i, blk in enumerate(self.body.layers[: self.layer + 1]):
            at, mlp = blk.self_attn, blk.mlp
            h = self.rms(x, blk.input_layernorm.weight)
            x = x + at.o_proj(self.attend(at.q_proj(h), at.k_proj(h), at.v_proj(h)))
            h = self.rms(x, blk.post_attention_layernorm.weight)
            hidden = F.silu(mlp.gate_proj(h)) * mlp.up_proj(h)
            if i == self.layer:
                return hidden
            x = x + mlp.down_proj(hidden)
        raise IndexError(self.layer)


def load(name: str) -> torch.nn.Module:
    """The model in fp32, in eval mode."""
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return model.eval()


def hf_activations(model: torch.nn.Module, layer: int, ids: torch.Tensor) -> torch.Tensor:
    """The MLP activations of ``layer`` from HF's own forward pass, for the agreement check."""
    if hasattr(model, "transformer"):
        write = model.transformer.h[layer].mlp.c_proj
    else:
        write = model.model.layers[layer].mlp.down_proj
    seen: list[torch.Tensor] = []
    handle = write.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    with torch.no_grad():
        model(ids)
    handle.remove()
    return seen[0]


def median_seconds(fn, sync=lambda: None) -> float:
    """The median wall time of ``REPS`` calls of ``fn``, after one warm-up call."""
    fn()
    sync()
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def agreement(ref: torch.Tensor, got) -> dict[str, float]:
    """How near ``got`` is to the fp32 activations ``ref``.

    The relative RMS error is over all values. The worst correlation is over 64 neurons
    spread across the layer. The overlap is of the top 20 tokens of neuron 0.
    """
    x = torch.as_tensor(np.asarray(got), dtype=torch.float32).reshape(ref.shape)
    step = max(1, ref.shape[-1] // 64)
    corrs = [
        float(torch.corrcoef(torch.stack([ref[..., n].flatten(), x[..., n].flatten()]))[0, 1])
        for n in range(0, ref.shape[-1], step)
    ]
    top = lambda t: set(t[..., 0].flatten().topk(20).indices.tolist())  # noqa: E731
    err = (ref - x).flatten()
    return {
        "rel_rms_err": float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
        "worst_neuron_corr": min(corrs),
        "top20_overlap": len(top(ref) & top(x)) / 20,
    }


def torch_times(net: ToLayer, ids: torch.Tensor, ref: torch.Tensor) -> dict:
    """Times on the CPU and on MPS, and how near MPS fp16 is to fp32."""
    out = {"torch_cpu_fp32_s": median_seconds(lambda: net(ids))}
    net, ids = net.to("mps"), ids.to("mps")
    out["torch_mps_fp32_s"] = median_seconds(lambda: net(ids), torch.mps.synchronize)
    net = net.half()
    out["torch_mps_fp16"] = {
        "s": median_seconds(lambda: net(ids), torch.mps.synchronize),
        **agreement(ref, net(ids).float().cpu()),
    }
    return out


def convert(net: ToLayer, ids: torch.Tensor, path: Path) -> float:
    """Trace ``net``, convert it to an fp16 Core ML package at ``path``, and return the time."""
    traced = torch.jit.trace(net, ids, check_trace=False)
    start = time.perf_counter()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=ids.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="acts")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    seconds = time.perf_counter() - start
    model.save(str(path))
    return seconds


def coreml_times(path: Path, ids: torch.Tensor, ref: torch.Tensor) -> dict:
    """For each compute-unit setting: load time, run time and agreement with fp32."""
    feed = {"ids": ids.numpy().astype(np.int32)}
    out = {}
    for unit in UNITS:
        start = time.perf_counter()
        model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit))
        load_s = time.perf_counter() - start
        out[unit] = {
            "load_s": load_s,
            "s": median_seconds(lambda m=model: m.predict(feed)),
            **agreement(ref, model.predict(feed)["acts"]),
        }
    return out


def ane_plan(path: Path) -> dict[str, int]:
    """How many operations Core ML puts on each device when the ANE is permitted."""
    from coremltools.models.compute_plan import MLComputePlan

    compiled = ct.utils.compile_model(str(path), str(path.with_suffix(".mlmodelc")))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts: Counter[str] = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            counts[type(usage.preferred_compute_device).__name__.removesuffix("ComputeDevice")] += 1
    return dict(counts)


def main(name: str, layer: int) -> None:
    """Print one JSON line of results for ``name`` at ``layer``."""
    torch.manual_seed(0)
    ids = torch.randint(0, 50000, (BATCH, SEQ))
    model = load(name)
    net = ToLayer(model, layer)
    with torch.no_grad():
        ref = net(ids)
        out = {"model": name, "layer": layer, "batch": [BATCH, SEQ]}
        hf = agreement(hf_activations(model, layer, ids), ref)
        out["rewrite_vs_hf_rel_err"] = hf["rel_rms_err"]
        out.update(torch_times(net, ids, ref))
    path = Path(tempfile.mkdtemp()) / "to_layer.mlpackage"
    with torch.no_grad():
        out["coreml_convert_s"] = convert(ToLayer(load(name), layer).eval(), ids, path)
    out["coreml"] = coreml_times(path, ids, ref)
    out["ane_plan_ops"] = ane_plan(path)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
