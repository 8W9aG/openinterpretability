"""Qualify a model for scans on Apple's Neural Engine (ANE), on a real corpus.

The device planner sends scans to the ANE only for a model in
``openinterp/harness/ane/qualified.py``. This script decides if a model can go there.
It runs a whole-document scan of ``--corpus`` for each of ``--layers``, in two ways, on
the same batches:

- on the ANE, through :class:`~openinterp.harness.ane.backend.AneScan`;
- in fp32 on Apple's GPU, which is the reference.

A model passes if every layer passes all four checks (:data:`CHECKS`):

- *top 20*: for every neuron and both poles, the scan's rescoring rule
  (``harness/scan_backend.py``) gives the exact values of the 20 strongest documents;
- *correlation*: each neuron's ANE activations correlate with fp32 over all tokens;
- *placement*: Core ML puts almost all operations on the ANE, not the CPU;
- *speed*: the ANE pass is faster than the fp32 pass on the CPU, which is the fallback.

It prints one JSON line per layer, and the verdict. Results go to
docs/analyses/ane-qualification.md.

    uv run python scripts/ane_qualify.py Qwen/Qwen2.5-0.5B --layers 4,20 --corpus corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import torch

from openinterp.cli import _load_runner
from openinterp.harness import scan_backend
from openinterp.harness.ane import cache
from openinterp.harness.ane.backend import AneScan
from openinterp.harness.scan import BATCH_TOKENS, batches, cut_pieces
from openinterp.reproduce import DEFAULT_TOLERANCE

CHECKS = {"top20": 1.0, "pearson_min": 0.9998, "ane_ops": 0.95, "speedup_over_cpu": 1.0}
CPU_BATCHES = 10  # batches that time the CPU fallback


def scan_both(ane: AneScan, runner, pieces, layer: int) -> dict:
    """Each piece's peak for each neuron and pole, on both backends, and correlation sums."""
    n, width = len(pieces), runner.mlp_width(layer)
    blank = lambda: torch.full((n, width), -torch.inf)  # noqa: E731
    peaks = {(b, p): blank() for b in ("ane", "fp32") for p in (1, -1)}
    sums, seconds, index = torch.zeros(6, width, dtype=torch.float64), Counter(), 0
    for batch in batches(runner, pieces, BATCH_TOKENS):
        start = time.perf_counter()
        y = ane.batch_acts(batch, layer)
        seconds["ane"] += time.perf_counter() - start
        start = time.perf_counter()
        with torch.no_grad():
            x = runner.layer_activations(batch.encoded, [layer])[layer].float().cpu()
        seconds["fp32"] += time.perf_counter() - start
        for row in range(len(batch.pieces)):
            mask = batch.mask[row]
            for name, acts in (("ane", y[row][mask]), ("fp32", x[row][mask])):
                for p in (1, -1):
                    peaks[(name, p)][index + row] = (p * acts).amax(0)
            a, b = x[row][mask].double(), y[row][mask].double()
            count = torch.full((width,), float(len(a)), dtype=torch.float64)
            sums += torch.stack(
                [count, a.sum(0), b.sum(0), (a * a).sum(0), (b * b).sum(0), (a * b).sum(0)]
            )
        index += len(batch.pieces)
    return {"peaks": peaks, "sums": sums, "seconds": dict(seconds)}


def top20_share(exact: torch.Tensor, ane: torch.Tensor) -> float:
    """Share of neurons whose rescored top-20 values match exact's, within the tolerance."""
    order = ane.argsort(dim=0, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(len(ane)).unsqueeze(1).expand_as(order))
    nth = ane.topk(scan_backend.RESCORE_RANK, dim=0).values[-1]
    line = nth - scan_backend.RESCORE_MARGIN * nth.abs() - 1e-3 * ane.amax(0).abs()
    top, cap = scan_backend.RESCORE_TOP, scan_backend.RESCORE_CAP
    chosen = (rank < top) | ((ane >= line) & (rank < cap))
    kept = torch.where(chosen, exact, torch.full_like(exact, -torch.inf)).topk(20, dim=0).values
    true = exact.topk(20, dim=0).values
    close = DEFAULT_TOLERANCE.atol * true[0].abs() + DEFAULT_TOLERANCE.rtol * true.abs()
    return float(((kept - true).abs() <= close).all(0).float().mean())


def pearson(sums: torch.Tensor) -> torch.Tensor:
    """Per-neuron Pearson correlation of the ANE's activations with fp32's."""
    n, sx, sy, sxx, syy, sxy = sums
    return (n * sxy - sx * sy) / ((n * sxx - sx * sx).sqrt() * (n * syy - sy * sy).sqrt())


def ane_ops(ane: AneScan, layer: int) -> float:
    """The smallest share of operations that Core ML puts on the ANE, over the layer's shapes."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    shares = []
    for path in ane.cache.root.glob(f"*/L{layer}/*/{cache.MODEL}"):
        meta = json.loads((path.parent / cache.META).read_text())["fields"]
        if meta["model"] != ane.model_name:
            continue
        plan = MLComputePlan.load_from_path(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        devices = Counter()
        for op in plan.model_structure.program.functions["main"].block.operations:
            if usage := plan.get_compute_device_usage_for_mlprogram_operation(op):
                devices[type(usage.preferred_compute_device).__name__] += 1
        total = sum(devices.values())
        shares.append(devices["MLNeuralEngineComputeDevice"] / total if total else 0.0)
    return min(shares, default=0.0)


def cpu_seconds_per_batch(model_name: str, pieces, layer: int) -> float:
    """Seconds per scan batch of the fp32 pass on the CPU, over :data:`CPU_BATCHES`."""
    runner = _load_runner(model_name, "cpu", "float32")
    seconds, count = 0.0, 0
    for batch in batches(runner, pieces, BATCH_TOKENS):
        start = time.perf_counter()
        with torch.no_grad():
            runner.layer_activations(batch.encoded, [layer])
        seconds, count = seconds + time.perf_counter() - start, count + 1
        if count == CPU_BATCHES:
            break
    return seconds / max(count, 1)


def qualify_layer(ane: AneScan, runner, model_name: str, pieces, layer: int) -> dict:
    """The four checks for one layer, and whether it passes them."""
    lengths = [int(m.sum()) for b in batches(runner, pieces, BATCH_TOKENS) for m in b.mask]
    if reason := ane.prepare(lengths, layer, BATCH_TOKENS):
        return {"model": model_name, "layer": layer, "passed": False, "reason": reason}
    both = scan_both(ane, runner, pieces, layer)
    n_batches = sum(1 for _ in batches(runner, pieces, BATCH_TOKENS))
    ane_per_batch = both["seconds"]["ane"] / n_batches
    corr = pearson(both["sums"])
    result = {
        "model": model_name,
        "layer": layer,
        "pieces": len(pieces),
        "top20": min(
            top20_share(both["peaks"][("fp32", p)], both["peaks"][("ane", p)]) for p in (1, -1)
        ),
        "pearson_min": float(corr.min()),
        "ane_ops": ane_ops(ane, layer),
        "speedup_over_cpu": cpu_seconds_per_batch(model_name, pieces, layer) / ane_per_batch,
        "ane_s_per_batch": ane_per_batch,
        "mps_fp32_s_per_batch": both["seconds"]["fp32"] / n_batches,
    }
    result["passed"] = all(result[k] >= v for k, v in CHECKS.items())
    return result


def main() -> None:
    """Qualify one model on each of ``--layers``; print a JSON line per layer and the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--layers", required=True, help="comma-separated, e.g. 4,20")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--max-docs", type=int, default=0, help="0: every document")
    parser.add_argument("--out", help="also append each JSON line to this file")
    args = parser.parse_args()
    with open(args.corpus) as lines:
        texts = [json.loads(line)["text"] for line in lines]
    texts = texts[: args.max_docs] if args.max_docs else texts
    runner = _load_runner(args.model, "mps", "float32")
    ane = AneScan(runner.model, args.model, runner.model_revision, cache.ConversionCache.default())
    pieces = list(cut_pieces(runner.tokenizer, texts, None, None))
    results = [
        qualify_layer(ane, runner, args.model, pieces, int(n)) for n in args.layers.split(",")
    ]
    for result in results:
        line = json.dumps(result)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as handle:
                handle.write(line + "\n")
    verdict = "QUALIFIED" if all(r["passed"] for r in results) else "NOT QUALIFIED"
    print(f"{verdict}: {args.model}, layers {args.layers}, {time.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
