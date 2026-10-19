"""Covariance between neurons: correlated partners and joint ablation.

A single neuron rarely acts alone. If neuron A's pole co-varies with neuron B, then
(a) a probe that "fails" on A may be handled by B, and (b) ablating A alone can show
no effect because B compensates. We therefore record A's partner neuron *poles* during
the scan, and the reviewer ablates A together with them.

Pole strengths are sparse and heavy-tailed (most tokens are exactly zero), so a global
Pearson correlation mostly measures agreement on "nothing is happening" and is driven
by a few large values. :class:`PartnerSearch` therefore ranks partners by the 95%
lower bound of their co-activation lift, which looks only at the regime where
redundancy matters and does not reward a handful of chance coincidences, and
reports a rank (Spearman) correlation per partner pole. Pearson on signed
activations is kept for reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from openinterp.core.schemas import CorrelatedNeuron, JointAblation, Pole
from openinterp.core.stats import spearman_columns
from openinterp.harness.hooks import Inputs, NeuronHookRunner, isolate_pole


class CovarianceAccumulator:
    """Streaming Pearson correlation between one target series and every layer neuron."""

    def __init__(self, width: int) -> None:
        self.n = 0
        self.sx = 0.0
        self.sxx = 0.0
        self.sy = torch.zeros(width, dtype=torch.float64)
        self.syy = torch.zeros(width, dtype=torch.float64)
        self.sxy = torch.zeros(width, dtype=torch.float64)

    def update(self, target: torch.Tensor, layer_acts: torch.Tensor) -> None:
        """``target`` is [n] pole strengths; ``layer_acts`` is [n, width] raw activations."""
        x, y = target.double(), layer_acts.double()
        self.n += x.numel()
        self.sx += float(x.sum())
        self.sxx += float((x * x).sum())
        self.sy += y.sum(0)
        self.syy += (y * y).sum(0)
        self.sxy += (x[:, None] * y).sum(0)

    def correlations(self) -> torch.Tensor:
        n = max(self.n, 1)
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        var_x = self.sxx / n - (self.sx / n) ** 2
        var_y = self.syy / n - (self.sy / n) ** 2
        denom = torch.sqrt(torch.clamp(var_y * var_x, min=0.0))
        return torch.where(denom > 1e-12, cov / denom, torch.zeros_like(cov)).clamp(-1, 1)


class PartnerSearch:
    """Stream a scan and find the neuron poles that co-activate with the target pole.

    Keeps, with fixed-seed randomness so results are reproducible:

    * a uniform reservoir sample of tokens (target strength + whole-layer activations),
      for partner base rates, 99th-percentile "active" thresholds and Spearman;
    * the ``top_k`` tokens where the target is strongest, which hold the target's
      active regime (it is rare, so a uniform sample would barely contain it).
    """

    def __init__(self, width: int, *, reservoir: int = 8192, top_k: int = 2048, seed: int = 0):
        self.pearson = CovarianceAccumulator(width)
        self.reservoir, self.top_k = reservoir, top_k
        self.generator = torch.Generator().manual_seed(seed)
        self.seen = 0
        self.res_target = torch.empty(0)
        self.res_rows = torch.empty(0, width, dtype=torch.float16)
        self.top_target = torch.empty(0)
        self.top_rows = torch.empty(0, width, dtype=torch.float16)

    def update(self, target: torch.Tensor, layer_acts: torch.Tensor) -> None:
        target, rows = target.float(), layer_acts.half()
        self.pearson.update(target, layer_acts)

        # Reservoir sampling (Algorithm R), vectorised per batch.
        fill = max(0, min(self.reservoir - self.res_target.numel(), target.numel()))
        self.res_target = torch.cat([self.res_target, target[:fill]])
        self.res_rows = torch.cat([self.res_rows, rows[:fill]])
        rest = torch.arange(fill, target.numel())
        if rest.numel():
            index = self.seen + rest + 1
            slots = (torch.rand(rest.numel(), generator=self.generator) * index).long()
            for i, slot in zip(
                rest[slots < self.reservoir].tolist(),
                slots[slots < self.reservoir].tolist(),
                strict=True,
            ):
                self.res_target[slot] = target[i]
                self.res_rows[slot] = rows[i]
        self.seen += target.numel()

        merged = torch.cat([self.top_target, target])
        keep = merged.topk(min(self.top_k, merged.numel())).indices
        self.top_target = merged[keep]
        self.top_rows = torch.cat([self.top_rows, rows])[keep]

    @staticmethod
    def _poles(rows: torch.Tensor) -> torch.Tensor:
        """[n, w] signed activations -> [n, 2w] pole strengths (positive, then negative)."""
        rows = rows.float()
        return torch.cat([rows.clamp(min=0), (-rows).clamp(min=0)], dim=1)

    def partners(
        self, exclude: int, active_threshold: float, *, k: int = 10, quantile: float = 0.99
    ) -> list[CorrelatedNeuron]:
        if self.res_target.numel() < 3:
            return []
        width = self.res_rows.shape[1]
        sample = self._poles(self.res_rows)
        thresholds = _column_quantile(sample, quantile)
        base = ((sample > thresholds).sum(0).double() + 1) / (sample.shape[0] + 2)

        active = self._poles(self.top_rows[self.top_target >= active_threshold])
        coactive = (active > thresholds).sum(0)
        n_active = active.shape[0]
        if n_active:
            lift = ((coactive.double() + 1) / (n_active + 2)) / base
            lower = wilson_lower(coactive.double(), n_active) / base
        else:  # the target never fired: no co-activation evidence, rank by Spearman alone
            lift = torch.ones(2 * width, dtype=torch.float64)
            lower = torch.zeros(2 * width, dtype=torch.float64)
        rho = spearman_columns(self.res_target, sample)
        pearson = self.pearson.correlations()

        candidates = [c for c in range(2 * width) if c % width != exclude]  # not itself
        order = sorted(
            candidates,
            key=lambda c: (-float(lower[c]), -abs(float(rho[c])), c),
        )[:k]
        return [
            CorrelatedNeuron(
                neuron_idx=c % width,
                pole="positive" if c < width else "negative",
                coactivation_lift=round(float(lift[c]), 4),
                coactivation_lift_lower=round(float(lower[c]), 4),
                coactive_tokens=int(coactive[c]),
                spearman=round(float(rho[c]), 6),
                pearson=round(float(pearson[c % width]), 6),
            )
            for c in order
        ]


def wilson_lower(successes: torch.Tensor, n: int, z: float = 1.96) -> torch.Tensor:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * torch.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / (1 + z * z / n)).clamp(min=0.0)


def _column_quantile(x: torch.Tensor, q: float, chunk: int = 1024) -> torch.Tensor:
    """Per-column nearest-rank quantile; chunked because torch.quantile caps input size."""
    k = max(1, min(x.shape[0], round(q * (x.shape[0] - 1)) + 1))
    return torch.cat(
        [x[:, i : i + chunk].kthvalue(k, dim=0).values for i in range(0, x.shape[1], chunk)]
    )


@torch.no_grad()
def _kl(runner: NeuronHookRunner, encoded: dict, fn: Callable | None, layer: int) -> float:
    mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
    clean = F.log_softmax(runner.run_with_interventions(encoded).float(), dim=-1)[mask]
    if fn is None:
        return 0.0
    ablated = F.log_softmax(runner.run_with_interventions(encoded, {layer: fn}).float(), dim=-1)[
        mask
    ]
    return max(0.0, float(F.kl_div(ablated, clean, log_target=True, reduction="batchmean")))


@torch.no_grad()
def joint_ablation(
    runner: NeuronHookRunner,
    prompts: Inputs,
    layer: int,
    neuron_idx: int,
    pole: Pole,
    partners: Sequence[int],
) -> JointAblation:
    """KL(clean ‖ ablated) for the pole alone, its partners alone, and all together.

    The pole is zero-ablated (its opposite pole is kept). Partners are mean-ablated over
    the prompts, which stays closer to their natural distribution than zeroing.
    """
    encoded = runner.encode(prompts)
    mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
    _, cache = runner.run_with_cache(encoded, [layer])
    partner_idx = torch.tensor(list(partners), dtype=torch.long)
    means = cache[layer][mask][:, partner_idx].mean(0) if len(partners) else None

    def ablate(target: bool, others: bool) -> Callable:
        def fn(act: torch.Tensor) -> torch.Tensor:
            act = act.clone()
            if target:
                act[..., neuron_idx] -= isolate_pole(act[..., neuron_idx], pole)
            if others and means is not None:
                act[..., partner_idx] = means.to(act.dtype)
            return act

        return fn

    kl_single = _kl(runner, encoded, ablate(True, False), layer)
    kl_partners = _kl(runner, encoded, ablate(False, True) if len(partners) else None, layer)
    kl_joint = _kl(runner, encoded, ablate(True, True), layer)
    ratio = (kl_joint - kl_partners) / kl_single if kl_single > 1e-9 else None
    return JointAblation(
        partners=list(partners),
        kl_single=round(kl_single, 9),
        kl_partners=round(kl_partners, 9),
        kl_joint=round(kl_joint, 9),
        redundancy_ratio=None if ratio is None else round(ratio, 6),
    )
