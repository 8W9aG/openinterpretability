"""Which model an explore draw studies: the smallest one below its coverage target.

One fully interpretable model is worth more than a little coverage of many. So the
models are ordered by size (MLP neurons, ``layers × width``). The smallest one whose
verified share is below the target gets most of the explore draws: the *focus*. A
neuron is verified when the latest record of one of its poles is VERIFIED. A pole that
the weights prove dead also counts (``openinterp.harness.dead``). When the focus passes
the target, the focus moves to the next model. The other models share
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
from openinterp.core.schemas import AuditRecord, ComponentIdentifier

# The focus moves on when this share of its neurons is verified.
DEFAULT_COVERAGE_TARGET = 0.8
# The share of explore draws left for the other models. ``[tool.openinterp]`` can set both.
DEFAULT_OTHERS_SHARE = 0.2


@dataclass(frozen=True)
class ModelCoverage:
    """How much of one model's MLP the registry and open claims cover."""

    model: str
    covered: int  # neurons with a record, a stopped scan or an open claim
    total: int | None  # every MLP neuron (layers × width); None when unknown
    verified: int = 0  # neurons with a VERIFIED pole

    @property
    def verified_share(self) -> float | None:
        """Verified share of the model's neurons, or None when its size is unknown."""
        return self.verified / self.total if self.total else None

    def __str__(self) -> str:
        """``319 covered, 62/36,864 verified (0.17%)``; no total when the size is unknown."""
        if not self.total:
            return f"{self.covered:,} covered, {self.verified:,} verified"
        return (
            f"{self.covered:,} covered, {self.verified:,}/{self.total:,} verified "
            f"({self.verified_share:.2%})"
        )


def model_config(model: str, download: bool = True) -> dict | None:
    """The model's ``config.json``, or None if it can't be read.

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
            return json.load(fh)
    except Exception:  # offline with nothing cached, or an unknown model
        return None


def model_dims(model: str, download: bool = True) -> tuple[int, int] | None:
    """(layers, MLP width) from the model's ``config.json``, or None if it can't be read.

    ``download`` is as for :func:`model_config`.
    """
    config = model_config(model, download)
    if config is None:
        return None
    return config_dims(config)


def model_heads(model: str, download: bool = True) -> tuple[int, int] | None:
    """(layers, attention heads per layer) from ``config.json``, or None if it can't be read."""
    config = model_config(model, download)
    if config is None:
        return None
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    heads = config.get("num_attention_heads") or config.get("n_head")
    return (layers, heads) if layers and heads else None


def config_dims(config: dict) -> tuple[int, int] | None:
    """(layers, MLP width) from a ``config.json``, or None if it does not give them."""
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    width = config.get("intermediate_size") or config.get("n_inner")
    if width is None and config.get("n_embd"):  # GPT-2: n_inner null means 4 × n_embd
        width = 4 * config["n_embd"]
    return (layers, width) if layers and width else None


def model_coverage(
    pool: Sequence[str],
    records: Iterable[AuditRecord],
    claims: Sequence[str],
    dead: Iterable[ComponentIdentifier] = (),
    stopped: Iterable[ComponentIdentifier] = (),
) -> dict[str, ModelCoverage]:
    """Coverage of each model in ``pool``.

    ``records`` are the latest record of each pole. A neuron is verified when one of
    them is VERIFIED. A pole in ``dead`` (proven dead by the weights) also counts. A pole
    in ``stopped`` (a scan stopped on it: dead or default-on) counts as covered. The
    size comes from the records (``model_num_layers × model_mlp_width``), else from the
    model's cached ``config.json``.
    """
    neurons: dict[str, set[tuple[int, int]]] = {m: set() for m in pool}
    verified: dict[str, set[tuple[int, int]]] = {m: set() for m in pool}
    dims: dict[str, tuple[int, int]] = {}
    for record in records:
        sub, ident = record.submission, record.submission.identifier
        if ident.model_name not in neurons or not isinstance(ident, NeuronIdentifier):
            continue
        neurons[ident.model_name].add((ident.layer, ident.neuron_idx))
        if record.verdict.status == "VERIFIED":
            verified[ident.model_name].add((ident.layer, ident.neuron_idx))
        if sub.model_num_layers and sub.model_mlp_width:
            dims[ident.model_name] = (sub.model_num_layers, sub.model_mlp_width)
    for ident in dead:
        if ident.model_name in neurons and isinstance(ident, NeuronIdentifier):
            neurons[ident.model_name].add((ident.layer, ident.neuron_idx))
            verified[ident.model_name].add((ident.layer, ident.neuron_idx))
    for ident in stopped:
        if ident.model_name in neurons and isinstance(ident, NeuronIdentifier):
            neurons[ident.model_name].add((ident.layer, ident.neuron_idx))
    coverage = {}
    for model in pool:
        claim = re.compile(rf"{re.escape(model_slug(model))}-L(\d+)-N(\d+)-")  # see claim_key
        for c in claims:
            neurons[model].update((int(m[1]), int(m[2])) for m in claim.finditer(c))
        layers_width = dims.get(model) or model_dims(model, download=False)
        total = layers_width[0] * layers_width[1] if layers_width else None
        coverage[model] = ModelCoverage(model, len(neurons[model]), total, len(verified[model]))
    return coverage


def focus_model(coverage: Iterable[ModelCoverage], target: float) -> str | None:
    """The smallest model of known size whose verified share is below ``target``, if any."""
    below = [c for c in coverage if c.total and c.verified / c.total < target]
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
