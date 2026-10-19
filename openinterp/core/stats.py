"""Statistics that account for dependence between observations.

Probes written by one reviewer are not independent: near-duplicates ("Many cities,
including Paris…" / "Many cities, including Rome…") share most of their information.
We cluster near-duplicate probes and treat each cluster as one observation, scored by
its mean pass rate. Effective counts feed the protocol's minimum-probe gates, the
balanced accuracy and the Beta posteriors behind significance and the registry's
false-discovery estimate.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_WORDS = re.compile(r"[a-z0-9']+")


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORDS.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def cluster_texts(texts: Sequence[str], threshold: float) -> list[int]:
    """Single-linkage clusters of texts whose word sets overlap by >= ``threshold``.

    Returns a cluster label per text, numbered in order of first appearance.
    """
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    words = [_words(t) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(words[i], words[j]) >= threshold:
                parent[find(j)] = find(i)
    labels: dict[int, int] = {}
    return [labels.setdefault(find(i), len(labels)) for i in range(len(texts))]


@dataclass(frozen=True)
class EffectiveCounts:
    raw: int  # probes
    clusters: int  # independent observations
    passed: float  # sum over clusters of the cluster's pass rate

    @property
    def rate(self) -> float | None:
        return self.passed / self.clusters if self.clusters else None


def effective_counts(
    texts: Sequence[str], passed: Sequence[bool], threshold: float
) -> EffectiveCounts:
    labels = cluster_texts(texts, threshold)
    groups: dict[int, list[bool]] = {}
    for label, ok in zip(labels, passed, strict=True):
        groups.setdefault(label, []).append(ok)
    return EffectiveCounts(
        raw=len(texts),
        clusters=len(groups),
        passed=sum(sum(g) / len(g) for g in groups.values()),
    )


def balanced_accuracy(specificity: EffectiveCounts, sensitivity: EffectiveCounts) -> float:
    rates = [r for r in (specificity.rate, sensitivity.rate) if r is not None]
    return sum(rates) / len(rates) if rates else 0.0


def p_balanced_above(
    specificity: EffectiveCounts,
    sensitivity: EffectiveCounts,
    threshold: float,
    *,
    samples: int = 4000,
    seed: int = 0,
) -> float:
    """Posterior P((spec + sens) / 2 > threshold) under Beta(1, 1) priors.

    Fixed ``seed`` keeps it deterministic, so identical records always score the same.
    """
    rng = random.Random(seed)

    def draw(c: EffectiveCounts) -> float:
        return rng.betavariate(1 + c.passed, 1 + c.clusters - c.passed)

    hits = sum((draw(specificity) + draw(sensitivity)) / 2 > threshold for _ in range(samples))
    return hits / samples


def expected_false_discoveries(p_useful: Sequence[float]) -> float:
    """Expected number of claimed discoveries that are false: sum of (1 - P(useful))."""
    return sum(1.0 - p for p in p_useful)


def rank_columns(x: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Ranks (0-based) down each column of ``x`` [n, c], averaging ties.

    Pole strengths are mostly exact zeros, so tie handling matters: every tied value
    gets the mean of the ranks it spans, as in ``scipy.stats.rankdata``.
    """
    n = x.shape[0]
    out = torch.empty_like(x, dtype=torch.float32)
    positions = torch.arange(n, dtype=torch.float32)[:, None]
    for start in range(0, x.shape[1], chunk):
        cols = x[:, start : start + chunk].float()
        values, order = cols.sort(dim=0)
        new = torch.ones_like(values, dtype=torch.bool)
        new[1:] = values[1:] != values[:-1]
        group = new.cumsum(0) - 1
        pos = positions.expand_as(values)
        sums = torch.zeros_like(values).scatter_add_(0, group, pos)
        counts = torch.zeros_like(values).scatter_add_(0, group, torch.ones_like(values))
        mean_rank = (sums / counts.clamp(min=1)).gather(0, group)
        out[:, start : start + chunk] = torch.empty_like(values).scatter_(0, order, mean_rank)
    return out


def spearman_columns(target: torch.Tensor, others: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Spearman's rho between ``target`` [n] and each column of ``others`` [n, c]."""
    ranks_t = rank_columns(target[:, None])[:, 0]
    t = ranks_t - ranks_t.mean()
    tt = (t * t).sum()
    rho = torch.zeros(others.shape[1])
    for start in range(0, others.shape[1], chunk):
        o = rank_columns(others[:, start : start + chunk])
        o = o - o.mean(0)
        denom = torch.sqrt(tt * (o * o).sum(0))
        part = (t[:, None] * o).sum(0) / denom
        rho[start : start + chunk] = torch.where(denom > 0, part, torch.zeros_like(part))
    return rho.clamp(-1, 1)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return 0.0
    return float(spearman_columns(x.float(), y.float()[:, None])[0])
