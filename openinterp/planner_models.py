"""Which model an explore draw studies: the smallest one below its coverage target.

One fully interpretable model is worth more than a little coverage of many. So the
models are ordered by size (MLP neurons, ``layers × width``). The smallest one whose
coverage is below the target gets most of the explore draws: the *focus*. When its
coverage passes the target, the focus moves to the next model. The other models share
the rest of the draws, ``∝ 1/√(1 + neurons covered)``, so sister draws and cross-model
comparisons still happen. Neurons within a model are still drawn at random, so each
model's statistics stay unbiased.

A neuron is covered when it has a record or an open PR claims it. A model of unknown
size can't be the focus. When no model can be, every model is weighted by coverage.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from openinterp.core.identifiers import NeuronIdentifier, model_slug
from openinterp.core.schemas import AuditRecord

# Coverage a model must pass before the focus moves on, and the share of explore draws
# left for the other models. ``[tool.openinterp]`` can set both.
DEFAULT_COVERAGE_TARGET = 0.05
DEFAULT_OTHERS_SHARE = 0.2


@dataclass(frozen=True)
class ModelCoverage:
    """How much of one model's MLP the registry and open claims cover."""

    model: str
    covered: int  # neurons with a record or an open claim
    total: int | None  # every MLP neuron (layers × width); None when unknown

    @property
    def share(self) -> float | None:
        """Covered share of the model's neurons, or None when its size is unknown."""
        return self.covered / self.total if self.total else None

    def __str__(self) -> str:
        """``319/36,864 (0.87%)``, or the covered count alone when the size is unknown."""
        return (
            f"{self.covered:,}/{self.total:,} ({self.share:.2%})"
            if self.total
            else f"{self.covered:,}"
        )


def model_dims(model: str, download: bool = True) -> tuple[int, int] | None:
    """(layers, MLP width) from the model's ``config.json``, or None if it can't be read.

    With ``download`` False, read only the Hugging Face cache: no network. The file is in
    the cache once the model has been used. Reading it never loads weights.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    try:
        path = hf_hub_download(model, "config.json") if download else None
        path = path or try_to_load_from_cache(model, "config.json")
        if not isinstance(path, str):  # not cached, or cached as missing
            return None
        with open(path) as fh:
            config = json.load(fh)
    except Exception:  # offline with nothing cached, or an unknown model
        return None
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    width = config.get("intermediate_size") or config.get("n_inner")
    if width is None and config.get("n_embd"):  # GPT-2: n_inner null means 4 × n_embd
        width = 4 * config["n_embd"]
    return (layers, width) if layers and width else None


def model_coverage(
    pool: Sequence[str], records: Iterable[AuditRecord], claims: Sequence[str]
) -> dict[str, ModelCoverage]:
    """Coverage of each model in ``pool``.

    The size comes from the records (``model_num_layers × model_mlp_width``), else from
    the model's cached ``config.json``.
    """
    neurons: dict[str, set[tuple[int, int]]] = {m: set() for m in pool}
    dims: dict[str, tuple[int, int]] = {}
    for record in records:
        sub, ident = record.submission, record.submission.identifier
        if ident.model_name not in neurons or not isinstance(ident, NeuronIdentifier):
            continue
        neurons[ident.model_name].add((ident.layer, ident.neuron_idx))
        if sub.model_num_layers and sub.model_mlp_width:
            dims[ident.model_name] = (sub.model_num_layers, sub.model_mlp_width)
    coverage = {}
    for model in pool:
        claim = re.compile(rf"{re.escape(model_slug(model))}-L(\d+)-N(\d+)-")  # see claim_key
        for c in claims:
            neurons[model].update((int(m[1]), int(m[2])) for m in claim.finditer(c))
        layers_width = dims.get(model) or model_dims(model, download=False)
        total = layers_width[0] * layers_width[1] if layers_width else None
        coverage[model] = ModelCoverage(model, len(neurons[model]), total)
    return coverage


def focus_model(coverage: Iterable[ModelCoverage], target: float) -> str | None:
    """The smallest model of known size whose coverage is below ``target``, if any."""
    below = [c for c in coverage if c.total and c.covered / c.total < target]
    return min(below, key=lambda c: (c.total, c.model)).model if below else None


def explore_weights(
    coverage: dict[str, ModelCoverage], target: float, others_share: float
) -> dict[str, float]:
    """Explore probability per model: ``1 − others_share`` for the focus model.

    The other models share ``others_share``, ``∝ 1/√(1 + neurons covered)``.
    """

    def by_coverage(models: Sequence[str]) -> dict[str, float]:
        weights = {m: 1.0 / math.sqrt(1 + coverage[m].covered) for m in models}
        return {m: w / sum(weights.values()) for m, w in weights.items()}

    focus = focus_model(coverage.values(), target)
    if focus is None or len(coverage) == 1:
        return by_coverage(list(coverage))
    others = by_coverage([m for m in coverage if m != focus])
    return {focus: 1.0 - others_share, **{m: others_share * w for m, w in others.items()}}
