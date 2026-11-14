# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.7.0", "transformers==4.46.3", "coremltools==9.0", "numpy"]
# ///
"""Try a scan's top examples from the ANE, measured again exactly, on a real corpus.

A whole-document scan ranks documents by their peak strength for a neuron pole and
records the top 20. This script makes those top 20 in two ways, for every neuron of one
layer and for each pole:

- *exact*: the forward pass in fp32 on MPS (:class:`ane_benchmark.ToLayer`);
- *screened*: the forward pass on the ANE gives the rank. The script then measures the
  candidates again in fp32 and keeps the best 20.

It reports the share of neurons whose 20 values agree with exact, inside the tolerance
of ``open-interpretability reproduce``. It does this for several ways to pick the candidates, and
also for the ANE's own values. Results go to docs/analyses/ane-feasibility.md.

The ANE needs fixed input shapes. So each document goes in the smallest of
:data:`SHAPES` that holds it, padded on the right. With causal attention, padding on the
right does not change the activations of real tokens. Run it with uv:

    HF_HUB_OFFLINE=1 uv run scripts/ane_corpus_screen.py openai-community/gpt2 5 corpus.jsonl
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import ane_benchmark as ab
import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

SHAPES = [(8, 256), (4, 512), (2, 1024)]  # (documents, tokens): 2,048 tokens each
TOP = 20  # top examples a scan records
KS = [20, 40, 60]  # fixed numbers of candidates
MARGINS = [0.02, 0.04, 0.06]  # candidates within this share of the ANE's 20th value
UNION = (60, 0.06, 120)  # the rule to use: top 60, or inside a 6% margin, at most 120


def read_docs(name: str, corpus: str) -> list[list[int]]:
    """Each document's token ids, at most the longest shape."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    with open(corpus) as lines:
        texts = [json.loads(line)["text"] for line in lines]
    return [tokenizer(text)["input_ids"][: SHAPES[-1][1]] for text in texts]


def batches(docs: list[list[int]]):
    """``(shape, document indices)``, with each document in the smallest shape that holds it."""
    shortest = 0
    for rows, seq in SHAPES:
        fits = [i for i, d in enumerate(docs) if shortest < len(d) <= seq]
        shortest = seq
        for start in range(0, len(fits), rows):
            yield (rows, seq), fits[start : start + rows]


def build(name: str, layer: int, shape: tuple[int, int]):
    """The fp32 model on MPS and the Core ML model on the ANE, for one input shape."""
    ab.BATCH, ab.SEQ = shape
    exact = ab.ToLayer(ab.load(name), layer).eval().to("mps")
    path = Path(tempfile.mkdtemp()) / "to_layer.mlpackage"
    with torch.no_grad():
        ab.convert(
            ab.ToLayer(ab.load(name), layer).eval(), torch.zeros(shape, dtype=torch.long), path
        )
    return exact, ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def run_batch(models, docs: list[list[int]], shape, fits) -> tuple[torch.Tensor, ...]:
    """fp32 and ANE activations of one batch, and the time each took."""
    exact, ane = models
    ids = torch.zeros(shape, dtype=torch.long)
    for row, i in enumerate(fits):
        ids[row, : len(docs[i])] = torch.tensor(docs[i])
    start = time.perf_counter()
    with torch.no_grad():
        a32 = exact(ids.to("mps")).float()
        torch.mps.synchronize()
    t_mps = time.perf_counter() - start
    start = time.perf_counter()
    a16 = ane.predict({"ids": ids.numpy().astype(np.int32)})["acts"]
    t_ane = time.perf_counter() - start
    a16 = torch.as_tensor(a16).float().reshape(a32.shape)
    return a32.cpu(), a16, torch.tensor([t_mps, t_ane])


class Peaks:
    """Each document's peak for each neuron and pole, from both backends, and correlation sums."""

    def __init__(self, n_docs: int, width: int):
        blank = lambda: torch.full((n_docs, width), -torch.inf)  # noqa: E731
        self.exact = {+1: blank(), -1: blank()}
        self.ane = {+1: blank(), -1: blank()}
        self.sums = torch.zeros(6, width, dtype=torch.float64)

    def add(self, doc: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add one document's activations: ``x`` exact, ``y`` from the ANE, ``[tokens, width]``."""
        for pole in (1, -1):
            self.exact[pole][doc] = (pole * x).amax(0)
            self.ane[pole][doc] = (pole * y).amax(0)
        x, y = x.double(), y.double()
        n = torch.full((x.shape[1],), float(len(x)), dtype=torch.float64)
        self.sums += torch.stack(
            [n, x.sum(0), y.sum(0), (x * x).sum(0), (y * y).sum(0), (x * y).sum(0)]
        )

    def pearson(self) -> torch.Tensor:
        """Per-neuron Pearson correlation of the ANE's activations with exact, over all tokens."""
        n, sx, sy, sxx, syy, sxy = self.sums
        return (n * sxy - sx * sy) / ((n * sxx - sx * sx).sqrt() * (n * syy - sy * sy).sqrt())


def scan(name: str, layer: int, docs: list[list[int]]) -> tuple[Peaks, torch.Tensor]:
    """Run every document on both backends. Return the peaks and the two total times."""
    peaks, times, models, current = None, torch.zeros(2), None, None
    for shape, fits in batches(docs):
        if shape != current:
            models, current = build(name, layer, shape), shape
        a32, a16, took = run_batch(models, docs, shape, fits)
        times += took
        peaks = peaks or Peaks(len(docs), a32.shape[-1])
        for row, doc in enumerate(fits):
            length = len(docs[doc])
            peaks.add(doc, a32[row, :length], a16[row, :length])
    return peaks, times


def agrees(kept: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Per neuron: are all ``TOP`` kept values inside the reproduce tolerance of exact's?"""
    true = exact.topk(TOP, dim=0).values
    return ((kept - true).abs() <= 1e-3 * true[0].abs() + 1e-2 * true.abs()).all(0)


def rescored(exact: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """The best ``TOP`` exact values among the chosen documents (``chosen`` is a mask)."""
    return torch.where(chosen, exact, torch.full_like(exact, -torch.inf)).topk(TOP, dim=0).values


def candidates(ane: torch.Tensor, k: int, margin: float, cap: int) -> torch.Tensor:
    """Mask of the ANE's top ``k`` documents, and of those inside ``margin``, at most ``cap``."""
    order = ane.argsort(dim=0, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(len(ane)).unsqueeze(1).expand_as(order))
    twentieth = ane.topk(TOP, dim=0).values[TOP - 1]
    inside = ane >= twentieth - margin * twentieth.abs() - 1e-3 * ane.amax(0).abs()
    return (rank < k) | (inside & (rank < cap))


def evaluate(exact: torch.Tensor, ane: torch.Tensor) -> dict:
    """For one pole: the share of neurons that agree, for each way to pick candidates."""
    width, n = exact.shape[1], len(exact)
    row = {"ane_alone": float(agrees(ane.topk(TOP, dim=0).values, exact).float().mean())}
    for k in KS:
        row[f"top{k}"] = float(
            agrees(rescored(exact, candidates(ane, k, 0.0, k)), exact).float().mean()
        )
    for m in MARGINS:
        chosen = candidates(ane, 0, m, n)
        row[f"margin{m}"] = float(agrees(rescored(exact, chosen), exact).float().mean())
    chosen = candidates(ane, *UNION)
    ok = agrees(rescored(exact, chosen), exact)
    counts = chosen.sum(0).float()
    row["union"] = float(ok.float().mean())
    row["union_failures"] = (~ok).nonzero().flatten().tolist()
    row["union_docs_mean"], row["union_docs_max"] = float(counts.mean()), float(counts.max())
    row["neurons"] = width
    return row


def main(name: str, layer: int, corpus: str) -> None:
    """Print one JSON line of results for ``name`` at ``layer`` on ``corpus``."""
    docs = read_docs(name, corpus)
    peaks, times = scan(name, layer, docs)
    pearson = peaks.pearson()
    out = {"model": name, "layer": layer, "docs": len(docs), "tokens": sum(map(len, docs))}
    out |= {"mps_fp32_s": float(times[0]), "ane_s": float(times[1])}
    out |= {"pearson_min": float(pearson.min()), "pearson_p1": float(pearson.quantile(0.01))}
    for pole, sign in (("+", 1), ("-", -1)):
        out[pole] = evaluate(peaks.exact[sign], peaks.ane[sign])
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
