"""One pass over a corpus that gathers everything an audit needs about a neuron pole.

For every non-padding token it records the pole strength (top examples and
histogram), the whole layer's activations (partner neuron poles: co-activation lift,
Spearman, Pearson) and the token id and position (confound profile).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from openinterp.core.schemas import (
    ActivatingExample,
    ConfoundProfile,
    CorrelatedNeuron,
    Pole,
    ReviewPolicy,
)
from openinterp.harness.confounds import ConfoundAccumulator
from openinterp.harness.covariance import PartnerSearch
from openinterp.harness.hooks import NeuronHookRunner, pole_strength


@dataclass
class ScanResult:
    examples: list[ActivatingExample]
    values: list[float]  # every non-padding token's pole strength
    correlated: list[CorrelatedNeuron]
    confounds: ConfoundProfile


def scan_neuron(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 20,
    batch_size: int = 8,
    n_correlated: int = 10,
    min_count: int = 3,
) -> ScanResult:
    if runner.tokenizer is None:
        raise ValueError("scan_neuron requires a tokenizer")
    partners = PartnerSearch(runner.mlp_width(layer))
    confounds = ConfoundAccumulator()
    examples: list[ActivatingExample] = []
    values: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = runner.encode(batch)
        _, cache = runner.run_with_cache(encoded, [layer])
        acts = cache[layer].float().cpu()  # [b, seq, d_mlp]
        strengths = pole_strength(acts[..., neuron_idx], pole)
        ids = encoded["input_ids"].cpu()
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool().cpu()
        positions = torch.arange(ids.shape[1]).expand_as(ids)

        partners.update(strengths[mask], acts[mask])
        confounds.update(ids[mask], positions[mask], strengths[mask])
        for row, text in enumerate(batch):
            row_vals = strengths[row][mask[row]].tolist()
            if not row_vals:
                continue
            values.extend(row_vals)
            peak = max(range(len(row_vals)), key=row_vals.__getitem__)
            examples.append(
                ActivatingExample(
                    text=text,
                    tokens=[runner.tokenizer.decode([i]) for i in ids[row][mask[row]].tolist()],
                    activations=[round(v, 5) for v in row_vals],
                    max_activation=row_vals[peak],
                    max_token_index=peak,
                )
            )

    examples.sort(key=lambda e: e.max_activation, reverse=True)
    # "Active" for co-activation uses the protocol's own definition for the target.
    active = ReviewPolicy().activation_threshold * max(values, default=0.0)
    return ScanResult(
        examples=examples[:k],
        values=values,
        correlated=partners.partners(neuron_idx, active, k=n_correlated) if active > 0 else [],
        confounds=confounds.finalize(runner.tokenizer, min_count),
    )
