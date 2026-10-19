"""Activation patching on single neuron poles.

Two directions are supported:

* **Denoising** (clean -> corrupted): run the corrupted prompt but inject the
  neuron's activation cached from the clean run. The normalised recovery of the
  clean behaviour measures *sufficiency*.
* **Noising** (corrupted -> clean): run the clean prompt but inject the activation
  from the corrupted run (a resample ablation). The normalised movement towards
  the corrupted behaviour measures *necessity*.

Only the requested pole is swapped; the opposite pole keeps its own value. With
``freeze_downstream=True`` the outputs of every later MLP are pinned to their
values from the unpatched run, which blocks MLP-mediated self-repair (the "hydra
effect") so that the measured effect is closer to the neuron's direct effect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch

from openinterp.core.exceptions import ShapeMismatchError
from openinterp.core.schemas import Pole
from openinterp.harness.hooks import Inputs, Intervention, NeuronHookRunner, replace_pole

Metric = Callable[[torch.Tensor], torch.Tensor]
Positions = Literal["all", "last"]


class NeuronRef(Protocol):
    layer: int
    neuron_idx: int
    pole: Pole


def token_logit_metric(token_id: int, position: int = -1) -> Metric:
    """Logit of ``token_id`` at ``position`` (averaged over the batch)."""
    return lambda logits: logits[:, position, token_id].mean()


def logit_diff_metric(correct_id: int, incorrect_id: int, position: int = -1) -> Metric:
    return lambda logits: (
        logits[:, position, correct_id] - logits[:, position, incorrect_id]
    ).mean()


def _normalised(patched: float, baseline: float, reference: float, eps: float = 1e-6) -> float:
    denom = reference - baseline
    return math.nan if abs(denom) < eps else (patched - baseline) / denom


@dataclass(frozen=True)
class PatchResult:
    """Outcome of patching a neuron pole from a source run into a target run.

    ``baseline`` is the target run's metric, ``reference`` the source run's, and
    ``patched`` the target run with the source activation injected.
    """

    baseline_metric: float
    reference_metric: float
    patched_metric: float
    logit_delta: torch.Tensor  # patched - baseline logits at the final position, [vocab]

    @property
    def metric_delta(self) -> float:
        return self.patched_metric - self.baseline_metric

    @property
    def recovery(self) -> float:
        """Fraction of the source-vs-target metric gap closed by the patch (NaN if no gap)."""
        return _normalised(self.patched_metric, self.baseline_metric, self.reference_metric)


@dataclass(frozen=True)
class AblationResult:
    per_source: list[PatchResult] = field(default_factory=list)

    @property
    def mean_metric_delta(self) -> float:
        return sum(r.metric_delta for r in self.per_source) / len(self.per_source)

    @property
    def mean_recovery(self) -> float:
        finite = [r.recovery for r in self.per_source if math.isfinite(r.recovery)]
        return sum(finite) / len(finite) if finite else math.nan


class ActivationPatcher:
    def __init__(self, runner: NeuronHookRunner) -> None:
        self.runner = runner

    def patch(
        self,
        source: Inputs,
        target: Inputs,
        neuron: NeuronRef,
        *,
        metric: Metric | None = None,
        positions: Positions = "all",
        freeze_downstream: bool = False,
    ) -> PatchResult:
        """Inject ``neuron``'s pole activation from the ``source`` run into the ``target`` run.

        If ``metric`` is omitted it defaults to the logit of the source run's
        top-1 next token at the final position.
        """
        runner = self.runner
        layer, idx, pole = neuron.layer, neuron.neuron_idx, neuron.pole
        downstream = list(range(layer + 1, runner.n_layers)) if freeze_downstream else []

        source_logits, source_cache = runner.run_with_cache(source, [layer])
        target_enc = runner.encode(target)
        target_logits, target_cache = runner.run_with_cache(target_enc, [layer])
        downstream_out = self._cache_outputs(target_enc, downstream)

        src_act, tgt_act = source_cache[layer], target_cache[layer]
        if positions == "all" and src_act.shape[:2] != tgt_act.shape[:2]:
            raise ShapeMismatchError(
                f"source {tuple(src_act.shape[:2])} and target {tuple(tgt_act.shape[:2])} "
                "differ in batch/sequence length; use positions='last' or align the prompts."
            )

        def inject(act: torch.Tensor) -> torch.Tensor:
            act = act.clone()
            if positions == "all":
                act[..., idx] = replace_pole(act[..., idx], src_act[..., idx], pole)
            else:
                act[:, -1, idx] = replace_pole(act[:, -1, idx], src_act[:, -1, idx], pole)
            return act

        frozen: dict[int, Intervention] = {
            i: (lambda _out, cached=cached: cached) for i, cached in downstream_out.items()
        }
        patched_logits = runner.run_with_interventions(target_enc, {layer: inject}, frozen)

        if metric is None:
            metric = token_logit_metric(int(source_logits[0, -1].argmax()))
        return PatchResult(
            baseline_metric=float(metric(target_logits)),
            reference_metric=float(metric(source_logits)),
            patched_metric=float(metric(patched_logits)),
            logit_delta=(patched_logits[0, -1] - target_logits[0, -1]).detach().cpu(),
        )

    def _cache_outputs(self, encoded, layers: Sequence[int]) -> dict[int, torch.Tensor]:
        if not layers:
            return {}
        cache: dict[int, torch.Tensor] = {}

        def recorder(layer: int) -> Intervention:
            def record(out: torch.Tensor) -> None:
                cache[layer] = out.detach().clone()

            return record

        self.runner.run_with_interventions(
            encoded, output_interventions={i: recorder(i) for i in layers}
        )
        return cache

    def denoise(self, clean: Inputs, corrupted: Inputs, neuron: NeuronRef, **kw) -> PatchResult:
        """Clean -> corrupted patch. ``recovery`` is the neuron's sufficiency."""
        return self.patch(clean, corrupted, neuron, **kw)

    def noise(self, clean: Inputs, corrupted: Inputs, neuron: NeuronRef, **kw) -> PatchResult:
        """Corrupted -> clean patch. ``recovery`` is the neuron's necessity."""
        return self.patch(corrupted, clean, neuron, **kw)

    def resample_ablation(
        self,
        target: Inputs,
        resample_sources: Sequence[Inputs],
        neuron: NeuronRef,
        *,
        metric: Metric | None = None,
        positions: Positions = "last",
        freeze_downstream: bool = False,
    ) -> AblationResult:
        """Replace the neuron pole in ``target`` with its value on each resample source.

        The default metric is fixed from the *target* run, so every source is scored
        against the same token.
        """
        if metric is None:
            target_logits = self.runner.run_with_interventions(target)
            metric = token_logit_metric(int(target_logits[0, -1].argmax()))
        results = [
            self.patch(
                src,
                target,
                neuron,
                metric=metric,
                positions=positions,
                freeze_downstream=freeze_downstream,
            )
            for src in resample_sources
        ]
        return AblationResult(per_source=results)
