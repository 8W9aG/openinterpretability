"""Measure which CPU options speed up the forward passes of a scan, and whether they agree (#3268).

Each option changes one thing against float32 on torch's default thread count:

* the thread count (``torch.set_num_threads``)
* the weights' dtype: float16 or bfloat16, which ``--dtype`` already offers
* GPT-2's ``NewGELU`` replaced by torch's fused ``gelu(approximate="tanh")``, which is the
  same formula computed in one vectorised kernel
* three compilers of the model to C++, timed without hooks, as an upper bound. They are
  ``torch.compile``, whose Inductor backend emits C++ with OpenMP on a CPU. TorchScript
  traces and freezes the model. AOTInductor compiles a shared library ahead of time.

The workload is ``runner.layer_activations``, the call that the scan makes for each batch:
one batch of 2,048 tokens (``scan.BATCH_TOKENS``) from ``corpus.jsonl``, in 512-token
windows. It stops at GPT-2's layer 5, the layer of ``perf_bench.py``'s pole, or at layer
11, the deepest. The options run in rounds, interleaved, because other jobs on the machine
change the timings. Each round times every option, and the rows give the median over the
rounds.

Agreement is against float32 on the default thread count. It uses the tolerance of
``reproduce`` (``openinterp.reproduce.Tolerance``). The largest difference over all MLP
activations of the layer must be at most ``atol`` times the largest float32 activation.

    uv run python scripts/cpu_acceleration.py --out docs/analyses/cpu-acceleration.json
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import click
import torch

MODEL = "openai-community/gpt2"
LAYERS = (5, 11)
WINDOW = 512
THREADS = (1, 2, 4, 8, 10)
ROUNDS = 5
PASSES = 3  # passes per option and round; the round's time is their median
SLOW = 10  # an option this many times slower than float32 is timed once, then dropped


def corpus_batch(tokenizer, tokens: int) -> dict[str, torch.Tensor]:
    """``tokens`` tokens of ``corpus.jsonl``, as rows of :data:`WINDOW` tokens."""
    ids: list[int] = []
    with open("corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            ids += tokenizer(json.loads(line)["text"])["input_ids"]
            if len(ids) >= tokens:
                break
    rows = torch.tensor(ids[:tokens]).view(-1, WINDOW)
    return {"input_ids": rows, "attention_mask": torch.ones_like(rows)}


def fuse_gelu(model) -> None:
    """Replace each GPT-2 block's ``NewGELU`` with torch's fused tanh GELU, in place."""
    for block in model.transformer.h:
        block.mlp.act = torch.nn.GELU(approximate="tanh")


def load(dtype: str, gelu: bool):
    """A runner of :data:`MODEL` on the CPU, in ``dtype``, with the fused GELU if ``gelu``."""
    from openinterp.cli import _load_runner

    runner = _load_runner(MODEL, "cpu", dtype, None)
    if gelu:
        fuse_gelu(runner.model)
    return runner


def options() -> dict[str, dict]:
    """``{name: settings}`` of every option, float32 on the default threads first."""
    default = torch.get_num_threads()
    out = {"float32": {"threads": default, "dtype": "float32", "gelu": False}}
    for n in THREADS:
        if n != default:
            out[f"float32, {n} threads"] = {"threads": n, "dtype": "float32", "gelu": False}
    out["float16"] = {"threads": default, "dtype": "float16", "gelu": False}
    out["bfloat16"] = {"threads": default, "dtype": "bfloat16", "gelu": False}
    out["fused GELU"] = {"threads": default, "dtype": "float32", "gelu": True}
    return out


def timed(fn) -> float:
    """The median wall time of :data:`PASSES` calls of ``fn``, in milliseconds."""
    times = []
    for _ in range(PASSES):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def agreement(acts: torch.Tensor, reference: torch.Tensor) -> dict:
    """How far ``acts`` is from ``reference``, against ``reproduce``'s tolerance."""
    from openinterp.reproduce import Tolerance

    diff = (acts.float() - reference).abs()
    scale = float(reference.abs().max())
    return {
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_over_max": round(float(diff.max()) / scale, 8),
        "within_reproduce_atol": float(diff.max()) <= Tolerance().atol * scale,
    }


def layer_runs(batch: dict, rounds: int) -> list[dict]:
    """Time and compare every option on both layers."""
    settings = options()
    runners = {name: load(s["dtype"], s["gelu"]) for name, s in settings.items()}
    rows = []
    for layer in LAYERS:
        times: dict[str, list[float]] = {name: [] for name in settings}
        acts: dict[str, torch.Tensor] = {}
        slow: set[str] = set()
        for _ in range(rounds):
            for name, s in settings.items():
                if name in slow:
                    continue
                torch.set_num_threads(s["threads"])
                runner = runners[name]
                with torch.inference_mode():
                    start = time.perf_counter()
                    acts[name] = runner.layer_activations(batch, [layer])[layer]  # warm
                    warm = (time.perf_counter() - start) * 1000
                    if times["float32"] and warm > SLOW * statistics.median(times["float32"]):
                        times[name].append(warm)  # one pass is enough to rule it out
                        slow.add(name)
                        continue
                    run = lambda r=runner, n=layer: r.layer_activations(batch, [n])  # noqa: E731
                    times[name].append(timed(run))
        reference = acts["float32"].float()
        base = statistics.median(times["float32"])
        for name, s in settings.items():
            ms = statistics.median(times[name])
            rows.append({
                "table": "layer_activations", "option": name, "layer": layer, **s,
                "median_ms": round(ms, 1), "min_ms": round(min(times[name]), 1),
                "max_ms": round(max(times[name]), 1), "speedup": round(base / ms, 3),
                **agreement(acts[name], reference),
            })  # fmt: skip
        torch.set_num_threads(settings["float32"]["threads"])
    return rows


class Logits(torch.nn.Module):
    """The model's logits only, with no cache, as one tensor that tracing and export take."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The logits of ``input_ids``."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits


def _torchscript(model, batch):
    """TorchScript: trace the model, freeze it, and optimise it for inference (C++ runtime)."""
    traced = torch.jit.trace(model, example_kwarg_inputs=batch, strict=False)
    frozen = torch.jit.optimize_for_inference(torch.jit.freeze(traced.eval()))
    return lambda b: frozen(**b)


def _aot_inductor(model, batch):
    """AOTInductor: export the model, compile it to a C++ shared library, and load it."""
    import tempfile

    exported = torch.export.export(model, args=(), kwargs=batch)
    path = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(Path(tempfile.mkdtemp()) / "gpt2.pt2")
    )
    loaded = torch._inductor.aoti_load_package(path)
    return lambda b: loaded(**b)


def _inductor(model, batch):
    """``torch.compile`` with its default backend, Inductor, which emits C++ with OpenMP."""
    compiled = torch.compile(model)
    return lambda b: compiled(**b)


COMPILERS = {
    "torch.compile (Inductor C++)": _inductor,
    "TorchScript, frozen": _torchscript,
    "AOTInductor (.so)": _aot_inductor,
}


def compiled_run(batch: dict, rounds: int) -> list[dict]:
    """Each compiler of the plain model against eager, without hooks (an upper bound).

    The scan's hooks stop the pass after the target layer by raising an exception, which a
    compiled graph can't do without a break. So this times the full model, which bounds
    what compiling could save. A compiler that fails gets a row with its error.
    """
    from transformers import AutoModelForCausalLM

    eager = Logits(AutoModelForCausalLM.from_pretrained(MODEL)).eval()
    with torch.inference_mode():
        reference = eager(**batch).float()
    runs = {"eager, no hooks": lambda b: eager(**b)}
    rows, firsts = [], {}
    for name, build in COMPILERS.items():
        start = time.perf_counter()
        try:
            run = build(eager, batch)
            with torch.inference_mode():
                out = run(batch)
        except Exception as exc:  # noqa: BLE001 - a compiler that fails is a result
            rows.append({"table": "compile", "option": name, "error": repr(exc)[:300]})
            continue
        firsts[name] = ((time.perf_counter() - start) * 1000, agreement(out, reference))
        runs[name] = run
    times: dict[str, list[float]] = {name: [] for name in runs}
    with torch.inference_mode():
        for _ in range(rounds):
            for name, run in runs.items():
                times[name].append(timed(lambda r=run: r(batch)))
    base = statistics.median(times["eager, no hooks"])
    for name, values in times.items():
        ms = statistics.median(values)
        row = {
            "table": "compile", "option": name, "median_ms": round(ms, 1),
            "min_ms": round(min(values), 1), "max_ms": round(max(values), 1),
            "speedup": round(base / ms, 3),
        }  # fmt: skip
        if name in firsts:
            row["build_and_first_call_ms"] = round(firsts[name][0], 1)
            row |= firsts[name][1]
        rows.append(row)
    return rows


def machine() -> dict:
    """What torch uses on this machine."""
    flags = {}
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.perflevel0.physicalcpu",
                    "hw.perflevel1.physicalcpu", "hw.optional.arm.FEAT_BF16",
                    "hw.optional.arm.FEAT_I8MM", "hw.optional.arm.FEAT_SME",
                    "hw.optional.neon_fp16"):  # fmt: skip
            done = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            flags[key] = done.stdout.strip()
    config = torch.__config__.show()
    blas = next((p.split("=")[1] for p in config.replace(",", " ").split()
                 if p.startswith("BLAS_INFO=")), None)  # fmt: skip
    return {
        "table": "machine", "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "blas": blas, "mkldnn": torch.backends.mkldnn.is_available(),
        "mkl": torch.backends.mkl.is_available(), "openmp": torch.backends.openmp.is_available(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "quantized_engines": list(torch.backends.quantized.supported_engines),
        "sysctl": flags, "load_average": list(__import__("os").getloadavg()),
    }  # fmt: skip


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--rounds", default=ROUNDS, show_default=True)
def main(out: Path, rounds: int) -> None:
    """Time every option, compare it with float32, and write the rows."""
    from transformers import AutoTokenizer

    from openinterp.harness.scan import BATCH_TOKENS

    batch = corpus_batch(AutoTokenizer.from_pretrained(MODEL), BATCH_TOKENS)
    rows = [machine()]
    for stage in (layer_runs, compiled_run):
        rows += stage(batch, rounds)
        rows[0]["load_average_after"] = list(__import__("os").getloadavg())
        out.write_text(json.dumps(rows, indent=1) + "\n")  # after each stage, in case one fails
    for r in rows[1:]:
        click.echo(json.dumps(r))


if __name__ == "__main__":
    main()
