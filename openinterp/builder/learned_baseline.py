"""The learned-basis baseline: random dictionary latents against random native neuron poles.

The native-neuron premise says a model's own MLP neurons are as explainable as the latents
of a sparse dictionary trained on the same layers. This compares the two under one
protocol (docs/PROTOCOL.md §1): for each dictionary, the share VERIFIED and the mean
balanced accuracy among its randomly selected latents, against the same among randomly
selected neuron poles of the same model on the layers the dictionary covers, scanned on
the same corpus with the same window (a scan's context changes what a unit is seen to do).

Each unit counts once, by its first reviewed experiment, and only if that experiment chose
it at random. Follow-ups (revisions, re-reviews) are left out so that neither arm gains
from being studied more. Neurons are also compared by pole: in a GELU MLP (GPT-2) the
negative pole is the activation's dip, at most 0.17 and reached at a pre-activation of
about −0.75. It sits in the dip on most tokens, so a probe's peak nearly always clears
the activity line and no hard negative passes. There the baseline's native arm is
positive poles (``audit --random --pole positive``), and :attr:`Comparison.p_positive`
is the test that matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb

from openinterp.harness.dictionaries import DICTIONARIES

FIRST_AUDITS = """
SELECT kind, model_name, layer, pole, dictionary, status, selection, balanced_accuracy,
       corpus_sha256, window_tokens,
       dictionary_explained_variance, dictionary_mean_l0, dictionary_dead_share
FROM r
WHERE kind IN ('mlp_neuron_pole', 'dictionary_feature') AND status IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY slug ORDER BY reviewed_at, experiment_id) = 1
"""


@dataclass(frozen=True)
class Arm:
    units: int
    verified: int
    mean_balanced_accuracy: float | None

    @property
    def share(self) -> float | None:
        return self.verified / self.units if self.units else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.verified, self.units) if self.units else None


@dataclass(frozen=True)
class Comparison:
    model_name: str
    dictionary: str
    layers: tuple[int, ...]
    corpus_sha256: str | None
    window_tokens: int | None  # None: whole documents
    features: Arm
    neurons: Arm
    positive_poles: Arm
    negative_poles: Arm
    explained_variance: float | None  # mean over the latents' scans
    mean_l0: float | None
    dead_share: float | None  # latents a random draw can't audit, so not in ``features``

    @property
    def p_value(self) -> float | None:
        """Two-sided Fisher exact test of equal shares VERIFIED, latents against every
        neuron pole."""
        return _fisher(self.features, self.neurons)

    @property
    def p_positive(self) -> float | None:
        """The same against positive poles only: the comparison for a GELU MLP, whose
        negative poles no hard negative can pass (docs/PROTOCOL.md §1)."""
        return _fisher(self.features, self.positive_poles)


def _fisher(a: Arm, b: Arm) -> float | None:
    if not (a.units and b.units):
        return None
    return fisher_exact(a.verified, a.units, b.verified, b.units)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p of Fisher's exact test on the 2×2 table (k1, n1−k1; k2, n2−k2)."""
    total, successes = n1 + n2, k1 + k2

    def prob(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, successes - k) / math.comb(total, successes)

    observed = prob(k1)
    lo, hi = max(0, successes - n2), min(n1, successes)
    p = sum(q for q in map(prob, range(lo, hi + 1)) if q <= observed * (1 + 1e-9))
    return min(1.0, p)


def _arm(rows: list[dict]) -> Arm:
    accuracies = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
    return Arm(
        units=len(rows),
        verified=sum(r["status"] == "VERIFIED" for r in rows),
        mean_balanced_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
    )


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def learned_baseline(con: duckdb.DuckDBPyConnection) -> list[Comparison]:
    """One comparison per dictionary and scan setting (corpus, window) with a randomly
    selected latent, over view ``r``."""
    cur = con.execute(FIRST_AUDITS)
    cols = [d[0] for d in cur.description]
    first = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    random = [r for r in first if r["selection"] == "random"]
    features: dict[tuple, list[dict]] = {}
    for r in random:
        if r["kind"] == "dictionary_feature":
            scan = (r["model_name"], r["corpus_sha256"], r["window_tokens"])
            features.setdefault((*scan, r["dictionary"]), []).append(r)
    comparisons = []
    for (*scan, name), latents in sorted(features.items(), key=lambda kv: str(kv[0])):
        spec = DICTIONARIES.get(name)
        layers = spec.layers if spec else tuple(sorted({r["layer"] for r in latents}))
        neurons = [
            r
            for r in random
            if r["kind"] == "mlp_neuron_pole"
            and (r["model_name"], r["corpus_sha256"], r["window_tokens"]) == tuple(scan)
            and r["layer"] in layers
        ]
        comparisons.append(
            Comparison(
                model_name=scan[0],
                dictionary=name,
                layers=layers,
                corpus_sha256=scan[1],
                window_tokens=scan[2],
                features=_arm(latents),
                neurons=_arm(neurons),
                positive_poles=_arm([r for r in neurons if r["pole"] == "positive"]),
                negative_poles=_arm([r for r in neurons if r["pole"] == "negative"]),
                explained_variance=_mean(r["dictionary_explained_variance"] for r in latents),
                mean_l0=_mean(r["dictionary_mean_l0"] for r in latents),
                dead_share=_mean(r["dictionary_dead_share"] for r in latents),
            )
        )
    return comparisons
