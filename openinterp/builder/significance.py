"""How conclusive an experiment's result is, in either direction.

A null result ("this hypothesis does not explain the neuron") is as valuable as a
positive one when the evidence is strong. So experiments are ranked by
*conclusiveness*, not by whether they passed.

Probe outcomes give two binomial samples: hard negatives correctly inactive
(specificity) and positive holdouts correctly active (sensitivity). Near-duplicate
probes are clustered and count once (:mod:`openinterp.core.stats`). Take uniform
Beta(1, 1) priors. Then the posterior probability that the hypothesis is useful is

    P(useful) = P((specificity + sensitivity) / 2 > threshold)

where ``threshold`` is the protocol's rejection line (balanced accuracy 0.6). We
estimate it by Monte Carlo with a fixed seed, so builds stay deterministic.

    conclusiveness = |2 · P(useful) − 1|      0 = no information, 1 = settled

The direction is "supports" when P(useful) ≥ 0.5 and "refutes" otherwise. More
probes sharpen the posterior, so a well-probed rejection outranks a thinly-probed
verification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openinterp.builder.disk_memo import DiskMemo
from openinterp.core import stats
from openinterp.core.schemas import AuditRecord, ReviewPolicy
from openinterp.core.stats import EffectiveCounts, effective_counts, p_balanced_above

# Conclusiveness at or above these is "strong" or "moderate" evidence, else "weak". The
# site's badges and the home page's conclusive-null count read them from here.
STRONG_EVIDENCE = 0.9
MODERATE_EVIDENCE = 0.5


def evidence_strength(conclusiveness: float) -> str:
    """The word for how settled a result is: strong, moderate or weak."""
    if conclusiveness >= STRONG_EVIDENCE:
        return "strong"
    return "moderate" if conclusiveness >= MODERATE_EVIDENCE else "weak"


@dataclass(frozen=True)
class Significance:
    """How conclusive one experiment is, and in which direction."""

    p_useful: float
    conclusiveness: float
    direction: Literal["supports", "refutes"]
    negatives: EffectiveCounts
    holdouts: EffectiveCounts

    @property
    def label(self) -> str:
        """The strength and direction as words, for example "strong evidence against"."""
        strength = evidence_strength(self.conclusiveness)
        return f"{strength} evidence {'for' if self.direction == 'supports' else 'against'}"


def significance(record: AuditRecord) -> Significance:
    """How conclusive ``record``'s result is (see the module docstring)."""
    verdict = record.verdict
    policy = verdict.policy if verdict and verdict.policy else ReviewPolicy()
    results = verdict.adversarial_test_results if verdict else []

    def counts(kind: str) -> EffectiveCounts:
        rows = [r for r in results if r.kind == kind and r.measured]
        return effective_counts(
            [r.prompt for r in rows], [r.passed for r in rows], policy.probe_cluster_threshold
        )

    negatives, holdouts = counts("hard_negative"), counts("positive_holdout")
    p_useful = _p_useful(negatives, holdouts, policy.reject_below_balanced_accuracy)
    return Significance(
        p_useful=round(p_useful, 4),
        conclusiveness=round(abs(2 * p_useful - 1), 4),
        direction="supports" if p_useful >= 0.5 else "refutes",
        negatives=negatives,
        holdouts=holdouts,
    )


# p_balanced_above samples 4,000 draws per record, so its results are kept on disk. The
# version is the code that computes them: a change to it starts a new memo.
P_USEFUL = DiskMemo("p-useful", Path(stats.__file__).read_bytes())


def _p_useful(negatives: EffectiveCounts, holdouts: EffectiveCounts, threshold: float) -> float:
    """Return :func:`p_balanced_above` for these counts, from the memo when it has it."""
    key = json.dumps(
        [negatives.passed, negatives.clusters, holdouts.passed, holdouts.clusters, threshold]
    )
    found = P_USEFUL.get(key)
    if found is None:
        found = p_balanced_above(negatives, holdouts, threshold)
        P_USEFUL.put(key, found)
    return found
