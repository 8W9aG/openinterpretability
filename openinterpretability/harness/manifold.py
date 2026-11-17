"""Measure the multi-dimensional feature check (``core.manifold.ManifoldCheck``).

The axes are the audited pole and its top partner poles. Each axis is divided by its
peak over every token of the top examples. So a partner with large activations doesn't
fill the first component. Unit variance would be worse: it would blow a partner that is
silent on a family, with tiny jitter, up into a full axis.

* **The activating set.** Each top example is run again, and the poles are read at the
  audited pole's peak.
* **The families.** Each word goes into every carrier sentence of
  :data:`openinterpretability.harness.specificity.CARRIERS`. A pole's strength on the word is its
  peak over the word's tokens, as a mean over the carriers.

A circle's score follows Engels et al. (2024). Their first component is often an
intensity direction and the circle lies in components 2 and 3. So the score takes the
better of the planes 1–2 and 2–3. The p-value repeats that choice on each shuffled
order, so it pays for the choice.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

import torch

from openinterpretability.core.manifold import FamilyManifold, ManifoldCheck, ManifoldPole
from openinterpretability.core.schemas import (
    ACTIVATION_THRESHOLD,
    ActivatingExample,
    CorrelatedNeuron,
)
from openinterpretability.core.stats import spearman
from openinterpretability.harness.hooks import NeuronHookRunner, pole_strength
from openinterpretability.harness.specificity import CARRIERS

PARTNERS = 5  # partner poles beside the audited pole
SHOWN = 3  # principal components whose explained variance is recorded
SHUFFLES = 999  # shuffled orders for each p-value
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "weekdays": ("circle", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                            "Saturday", "Sunday")),
    "months": ("circle", ("January", "February", "March", "April", "May", "June", "July",
                          "August", "September", "October", "November", "December")),
    "numbers": ("line", ("one", "two", "three", "four", "five", "six", "seven", "eight",
                         "nine", "ten", "eleven", "twelve")),
}  # fmt: skip


def axes(neuron_idx: int, pole, partners: Sequence[CorrelatedNeuron]) -> list[ManifoldPole]:
    """The audited pole, then its top :data:`PARTNERS` partner poles."""
    own = ManifoldPole(neuron_idx=neuron_idx, pole=pole)
    return [own, *(ManifoldPole(neuron_idx=c.neuron_idx, pole=c.pole) for c in partners)][
        : 1 + PARTNERS
    ]


def _strengths(acts: torch.Tensor, poles: Sequence[ManifoldPole]) -> torch.Tensor:
    """``[seq, len(poles)]``: each pole's strength at each token, from layer ``acts``."""
    columns = [pole_strength(acts[:, p.neuron_idx], p.pole) for p in poles]
    return torch.stack(columns, dim=1).float().cpu()


def _layer(runner: NeuronHookRunner, text: str, layer: int) -> torch.Tensor:
    """``[seq, width]``: the MLP hidden activations of ``layer`` on ``text``."""
    return runner.layer_activations(text, [layer])[layer][0]


def word_vector(runner, layer: int, poles, word: str) -> torch.Tensor | None:
    """Each pole's peak over ``word``'s tokens, as a mean over the carriers.

    ``None`` when no carrier measures a token of the word.
    """
    found = []
    for carrier in CARRIERS:
        text, start = carrier.format(word), carrier.index("{}")
        offsets = runner.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        values = _strengths(_layer(runner, text, layer), poles)
        held = [i for i, (a, b) in enumerate(offsets) if a < start + len(word) and b > start]
        if len(offsets) == len(values) and held:
            found.append(values[held].max(dim=0).values)
    return torch.stack(found).mean(dim=0) if found else None


def peak_vector(runner, layer: int, poles, example: ActivatingExample, skip_first: bool):
    """The poles at the audited pole's peak, and each pole's peak over the example.

    The example's text is run again.
    """
    values = _strengths(_layer(runner, example.text, layer), poles)
    start = int(skip_first and len(values) > 1)
    return values[start + int(values[start:, 0].argmax())], values[start:].max(dim=0).values


def scaled(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``rows`` with each pole divided by its scale; a pole with no scale is 0."""
    return torch.where(scales > 0, rows / scales.clamp_min(1e-12), torch.zeros_like(rows))


def pca(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The coordinates, the variances and the unit components of the centred ``rows``."""
    z = rows.double() - rows.double().mean(dim=0)
    u, s, vt = torch.linalg.svd(z, full_matrices=False)
    return u * s, s**2, vt


def ratios(variances: torch.Tensor) -> list[float]:
    """The first :data:`SHOWN` shares of variance, or none when nothing varies."""
    total = float(variances.sum())
    return [round(float(v) / total, 4) for v in variances[:SHOWN]] if total > 0 else []


def participation_ratio(variances: torch.Tensor) -> float:
    """``(Σλ)² / Σλ²``: the effective number of dimensions; 0 when nothing varies."""
    squares = float((variances**2).sum())
    return round(float(variances.sum()) ** 2 / squares, 4) if squares > 0 else 0.0


def circle_score(coords: torch.Tensor, places: Sequence[int], n: int) -> float:
    """The mean resultant length of angle minus cycle place, in the better direction."""
    angles = [math.atan2(float(y), float(x)) for x, y in coords]
    cycle = [2 * math.pi * k / n for k in places]
    forward = abs(sum(cmath.exp(1j * (a - c)) for a, c in zip(angles, cycle, strict=True)))
    backward = abs(sum(cmath.exp(1j * (a + c)) for a, c in zip(angles, cycle, strict=True)))
    return max(forward, backward) / len(angles)


def _best(shape: str, coords: torch.Tensor, order: Sequence[int]) -> tuple[float, list[int]]:
    """The best score over the allowed components, and those components (1-based).

    On a tie the first components win. With no components the score is 0.
    """
    if shape == "circle":
        planes = [[0, 1], [1, 2]][: max(coords.shape[1] - 1, 0)]
        found = [(circle_score(coords[:, p], order, len(order)), p) for p in planes]
    else:
        values = torch.tensor(order, dtype=torch.float64)
        found = [
            (abs(spearman(coords[:, k], values)), [k]) for k in range(min(SHOWN, coords.shape[1]))
        ]
    score, picked = max(found, key=lambda f: f[0], default=(0.0, []))
    return score, [k + 1 for k in picked]


def family_manifold(
    family: str, rows: torch.Tensor, threshold: float, scales: torch.Tensor | None = None
) -> FamilyManifold:
    """The PCA and shape of one family's ``rows`` (members in order, poles as columns).

    ``threshold`` is in the audited pole's units. The PCA sees ``rows`` divided by
    ``scales`` (:func:`scaled`), when given.
    """
    shape = FAMILIES[family][0]
    coords, variances, vt = pca(rows if scales is None else scaled(rows, scales))
    order = list(range(len(rows)))
    if not float(variances.sum()) > 0:  # nothing varies: no shape and no components
        coords = coords[:, :0]
    score, components = _best(shape, coords, order)
    rng = torch.Generator().manual_seed(len(rows))  # fixed: a record always replays
    beaten = sum(
        _best(shape, coords, torch.randperm(len(rows), generator=rng).tolist())[0] >= score - 1e-9
        for _ in range(SHUFFLES)
    )
    share = float((vt[[c - 1 for c in components], 0] ** 2).sum()) if components else 0.0
    plane = sorted((float(variances[c - 1]) for c in components), reverse=True)
    round_ = plane[1] / plane[0] if shape == "circle" and len(plane) == 2 and plane[0] > 0 else None
    return FamilyManifold(
        family=family, shape=shape, members=len(rows),
        active_members=int((rows[:, 0] >= threshold).sum()),
        explained_variance=ratios(variances), components=components,
        score=round(min(score, 1.0), 4), p_value=round((1 + beaten) / (1 + SHUFFLES), 4),
        pole_share=round(min(share, 1.0), 4),
        roundness=None if round_ is None else round(round_, 4),
    )  # fmt: skip


def manifold_check(
    runner: NeuronHookRunner,
    layer: int,
    poles: list[ManifoldPole],
    examples: Sequence[ActivatingExample],
    skip_first: bool = False,
) -> ManifoldCheck:
    """Measure the check for ``poles[0]`` and its partners in ``layer``."""
    max_act = max((e.max_activation for e in examples), default=0.0)
    measured = [peak_vector(runner, layer, poles, e, skip_first) for e in examples]
    peaks = [p for p, _ in measured]
    scales = torch.stack([s for _, s in measured]).max(dim=0).values if measured else None
    variances = pca(scaled(torch.stack(peaks), scales))[1] if len(peaks) > 1 else torch.zeros(0)
    families = []
    for family, (_, words) in FAMILIES.items():
        vectors = [word_vector(runner, layer, poles, w) for w in words]
        if all(v is not None for v in vectors):
            rows = torch.stack(vectors)
            threshold = ACTIVATION_THRESHOLD * max_act
            families.append(family_manifold(family, rows, threshold, scales))
    return ManifoldCheck(
        poles=poles,
        activating_examples=len(peaks),
        activating_explained_variance=ratios(variances),
        participation_ratio=participation_ratio(variances),
        families=families,
    )
