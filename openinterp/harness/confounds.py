"""Confound control: how much of a pole's activation trivial explanations account for.

During the scan we accumulate, per token id and per position, the count, sum and sum
of squares of the pole strength. From that we derive:

* ``token_r2`` / ``position_r2``: variance explained by a group-mean model (between-
  group sum of squares over total sum of squares);
* ``frequency_correlation``: Spearman's rho, over tokens seen ``min_count`` times or
  more, between token frequency and mean strength. Rank-based, because mean strength
  per token is heavily skewed and the relationship of interest is monotonic;
* trivial *baselines* for the reviewer's probes: "active if any token is a trigger
  token" and "active if any position is a high-activation position".

A hypothesis that cannot match the best trivial baseline on the same probes adds
nothing beyond "this neuron fires on token X" (or "at position p").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openinterp.core.schemas import ConfoundProfile, TokenStat
from openinterp.core.stats import spearman

MAX_TRIGGER_TOKENS = 200


class ConfoundAccumulator:
    def __init__(self) -> None:
        self.token: dict[str, torch.Tensor] = {}
        self.position: dict[str, torch.Tensor] = {}
        self.total = torch.zeros(3, dtype=torch.float64)  # n, sum, sum of squares

    @staticmethod
    def _add(store: dict[str, torch.Tensor], keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        for name, weights in (
            ("n", torch.ones_like(values)),
            ("s", values),
            ("ss", values * values),
        ):
            counts = torch.bincount(keys, weights=weights)
            current = store.get(name, torch.zeros(0, dtype=torch.float64))
            if current.numel() < counts.numel():
                pad = torch.zeros(counts.numel() - current.numel(), dtype=torch.float64)
                current = torch.cat([current, pad])
            current[: counts.numel()] += counts
            store[name] = current

    def update(self, ids: torch.Tensor, positions: torch.Tensor, strengths: torch.Tensor) -> None:
        values = strengths.double()
        self._add(self.token, ids.long(), values)
        self._add(self.position, positions.long(), values)
        self.total += torch.tensor(
            [values.numel(), values.sum(), (values * values).sum()], dtype=torch.float64
        )

    def _r2(self, store: dict[str, torch.Tensor]) -> float:
        n, s, ss = (float(x) for x in self.total)
        if n == 0:
            return 0.0
        total = ss - s * s / n
        seen = store["n"] > 0
        between = float((store["s"][seen] ** 2 / store["n"][seen]).sum()) - s * s / n
        return min(1.0, max(0.0, between / total)) if total > 1e-12 else 0.0

    def finalize(self, tokenizer: Any, min_count: int = 3) -> ConfoundProfile:
        if not self.token:
            return ConfoundProfile(
                token_r2=0.0,
                position_r2=0.0,
                frequency_correlation=0.0,
                min_count=min_count,
                trigger_tokens=[],
                position_means=[],
            )
        n, s = self.token["n"], self.token["s"]
        frequent = (n >= min_count).nonzero().flatten()
        means = s[frequent] / n[frequent]
        order = means.argsort(descending=True)[:MAX_TRIGGER_TOKENS]
        triggers = [
            TokenStat(
                token=tokenizer.decode([int(frequent[i])]) if tokenizer else str(int(frequent[i])),
                token_id=int(frequent[i]),
                mean_strength=round(max(0.0, float(means[i])), 6),
                count=int(n[frequent[i]]),
            )
            for i in order.tolist()
        ]
        pn, ps = self.position["n"], self.position["s"]
        position_means = [
            round(float(ps[p] / pn[p]), 6) if pn[p] > 0 else 0.0 for p in range(pn.numel())
        ]
        return ConfoundProfile(
            token_r2=round(self._r2(self.token), 6),
            position_r2=round(self._r2(self.position), 6),
            frequency_correlation=round(spearman(n[frequent], means), 6),
            min_count=min_count,
            trigger_tokens=triggers,
            position_means=position_means,
        )


def baseline_predictions(
    profile: ConfoundProfile, tokenizer: Any, texts: Sequence[str], threshold: float
) -> dict[str, list[bool]]:
    """Predicted "active" for each text under each trivial baseline."""
    triggers = {t.token_id for t in profile.trigger_tokens if t.mean_strength >= threshold}
    hot_positions = {p for p, m in enumerate(profile.position_means) if m >= threshold}
    token_pred, position_pred = [], []
    for text in texts:
        ids = tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()
        token_pred.append(any(i in triggers for i in ids))
        position_pred.append(any(p in hot_positions for p in range(len(ids))))
    return {"token identity": token_pred, "position": position_pred}
