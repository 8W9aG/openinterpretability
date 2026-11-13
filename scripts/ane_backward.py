"""Can the ANE run a backward pass? A prototype on GPT-2: one neuron's input gradient.

The gradient is of one neuron's activation, summed over positions, with respect to the
input embeddings, through layers 0 to L. Input attribution needs such gradients
(``harness/input_attribution.py``). Core ML converts a forward graph, so the backward
pass is written out as one. ``torch.func.grad`` gives the gradient, and ``make_fx`` traces
it to plain aten operations. ``torch.export`` gives the graph that coremltools converts.

Three changes make that graph convert and run well (docs/analyses/ane-backward.md):

- LayerNorm is written out in plain operations. ``native_layer_norm`` returns its mean
  and inverse standard deviation too, and the gradient reads them. coremltools binds
  only the first output of an operation.
- GELU's ``exp`` gets an argument of at most 11. In ``forward.py``'s form, ``exp`` can
  overflow. The forward value is then 0, which is correct, but its gradient is inf/inf.
- The core aten decompositions rewrite gradient operations (``select_backward`` and
  others) as forward ones that coremltools knows.

The script compares the gradient on the ANE and on Core ML's GPU (both fp16) with torch
fp32 and fp16 on the CPU. It also reports the share of operations that Core ML puts on
the ANE. One JSON row per layer and neuron goes to ``--out``.

    uv run python scripts/ane_backward.py --layers 3,6 --neurons 100,2000 --out backward.jsonl
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import click
import numpy as np
import torch
from torch.func import grad

from openinterp.cli import _load_runner
from openinterp.harness.ane import chunks
from openinterp.harness.ane.forward import ToLayer

MODEL = "openai-community/gpt2"
SCALE = 256.0  # the loss scale: fp16 gradients of one neuron are small
EXP_LIMIT = 11.0  # exp(11) = 59,874, inside fp16's range
TIMED = 5  # passes timed for each backend


def gelu_for_gradients(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's GELU as ``forward.gelu_new`` writes it, with a finite gradient.

    Below the limit the result differs by at most |x| / (1 + e^11), less than 2e-5 |x|.
    """
    z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return x / (1.0 + torch.exp((-2.0 * z).clamp(max=EXP_LIMIT)))


class FromEmbeddings(torch.nn.Module):
    """GPT-2 from the input embeddings to one layer's MLP activations, in plain operations."""

    def __init__(self, net: ToLayer):
        super().__init__()
        self.net = net

    @staticmethod
    def layer_norm(norm: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm with one output, and ``x * x`` for its squares."""
        c = x - x.mean(-1, keepdim=True)
        return c * torch.rsqrt((c * c).mean(-1, keepdim=True) + norm.eps) * norm.weight + norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The MLP activations of ``net.layer``."""
        net = self.net
        for i in range(net.layer + 1):
            block = net.body.h[i]
            qkv = block.attn.c_attn(self.layer_norm(block.ln_1, x)).chunk(3, -1)
            x = x + block.attn.c_proj(net.attend(*qkv))
            hidden = gelu_for_gradients(block.mlp.c_fc(self.layer_norm(block.ln_2, x)))
            if i == net.layer:
                return hidden
            x = x + block.mlp.c_proj(hidden)
        raise AssertionError("unreachable")


def gradient_fn(module: FromEmbeddings, neuron: int):
    """The function from embeddings to the gradient of ``neuron``'s summed activation."""

    def target(x):
        return module(x)[..., neuron].sum() * SCALE

    return lambda x: grad(target)(x) / SCALE


def convert(fn, emb: torch.Tensor, dest: Path):
    """``fn`` as an fp16 Core ML model, saved to ``dest``; and how long the conversion took."""
    import coremltools as ct
    from torch._decomp import core_aten_decompositions, get_decompositions
    from torch.fx.experimental.proxy_tensor import make_fx

    start = time.perf_counter()
    table = {
        **core_aten_decompositions(),
        **get_decompositions([torch.ops.aten.native_layer_norm_backward]),
    }
    flat = make_fx(fn, decomposition_table=table)(emb)
    exported = torch.export.export(flat, (emb,)).run_decompositions({})
    model = ct.convert(
        exported,
        inputs=[ct.TensorType(name="emb", shape=emb.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="grad")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(dest))
    return time.perf_counter() - start


def placement(package: Path) -> float | None:
    """The share of operations that Core ML puts on the ANE, or None if it has no plan."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = package.with_suffix(".mlmodelc")
    ct.utils.compile_model(str(package), str(compiled))
    try:
        plan = MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except RuntimeError:  # e.g. error -7: Core ML can't build the plan
        return None
    where = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        if usage := plan.get_compute_device_usage_for_mlprogram_operation(op):
            where[type(usage.preferred_compute_device).__name__] += 1
    return where["MLNeuralEngineComputeDevice"] / max(sum(where.values()), 1)


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """Relative RMS error and cosine of ``got`` against ``ref``."""
    err = float((ref - got).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    cosine = float(torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0))
    return {"rel_error": err, "cosine": cosine}


def on_core_ml(package: Path, emb: torch.Tensor, unit: str, shape) -> tuple[torch.Tensor, float]:
    """The gradient from the Core ML model on ``unit``, and its milliseconds per pass."""
    import coremltools as ct

    model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, unit))
    feed = {"emb": emb.numpy().astype(np.float16)}
    out = torch.as_tensor(np.asarray(model.predict(feed)["grad"], dtype=np.float32))
    start = time.perf_counter()
    for _ in range(TIMED):
        model.predict(feed)
    return out.reshape(shape), (time.perf_counter() - start) / TIMED * 1000


def measure(module: FromEmbeddings, half: FromEmbeddings, neuron: int, emb: torch.Tensor) -> dict:
    """One neuron's gradient on each backend: its error against fp32, and its time."""
    fn = gradient_fn(module, neuron)
    ref = fn(emb)
    start = time.perf_counter()
    for _ in range(TIMED):
        fn(emb)
    row = {
        "cpu_fp32_ms": (time.perf_counter() - start) / TIMED * 1000,
        "cpu_fp16": compare(ref, gradient_fn(half, neuron)(emb.half()).float()),
    }
    work = Path(tempfile.mkdtemp(prefix="openinterp-backward-"))
    try:
        row["convert_s"] = convert(fn, emb, work / "grad.mlpackage")
    except Exception as exc:  # report every conversion failure, then go on
        return {**row, "error": f"conversion: {type(exc).__name__}: {str(exc)[:200]}"}
    row["ane_ops"] = placement(work / "grad.mlpackage")
    for unit in ("CPU_AND_NE", "CPU_AND_GPU"):
        try:
            got, ms = on_core_ml(work / "grad.mlpackage", emb, unit, ref.shape)
            row[unit] = {**compare(ref, got), "ms": ms}
        except Exception as exc:  # e.g. the ANE can't build the plan
            row[unit] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return row


@click.command()
@click.option("--layers", default="3", show_default=True, help="Comma-separated.")
@click.option("--neurons", default="100", show_default=True, help="Comma-separated.")
@click.option("--seq", type=int, default=128, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(layers, neurons, seq, out):
    """Compare a neuron's input gradient on the ANE with fp32, for each layer and neuron."""
    runner = _load_runner(MODEL, "cpu", "float32")
    vocab = runner.model.config.vocab_size
    ids = torch.randint(0, vocab, (1, seq), generator=torch.Generator().manual_seed(3))
    for layer in (int(n) for n in layers.split(",")):
        part = chunks.part(runner.model, MODEL, runner.model_revision, 0, layer)
        body, _, _ = chunks.parts(part)
        with torch.no_grad():
            emb = (body.wte(ids) + body.wpe(torch.arange(seq))).float()
        net = ToLayer(part, layer, 1, seq).eval()
        module = FromEmbeddings(net).eval().requires_grad_(False)
        half_net = copy.deepcopy(net).half()
        half_net.extra = {
            k: v.half() if v.is_floating_point() else v for k, v in half_net.extra.items()
        }
        half = FromEmbeddings(half_net).eval().requires_grad_(False)
        for neuron in (int(n) for n in neurons.split(",")):
            row = {"model": MODEL, "layer": layer, "neuron": neuron, "seq": seq}
            row.update(measure(module, half, neuron, emb))
            line = json.dumps(row)
            click.echo(line)
            with open(out, "a") as handle:
                handle.write(line + "\n")


if __name__ == "__main__":
    main()
