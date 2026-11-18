"""Rival readings of a unit, scored on the review's probes (issue #3197). Reported, not gated.

The proposer may write up to :data:`MAX_RIVALS` rival readings beside its hypothesis: other
explanations that fit the same top examples but predict differently elsewhere. The reviewer
labels each probe with what each rival predicts, as well as what the hypothesis predicts,
and writes some probes where they disagree. Each probe is measured once. Each reading is
then scored by its balanced accuracy on the measured probes, against whether each probe
was active.

The idea is from Macke, Gedon and Wahl, "Teaching AI to build scientific models": keep a
range of plausible models, not one best fit. Rivals force probes that separate readings,
not only probes that break one claim. The verdict does not use the comparison. Making the
surviving reading decide a verdict would be a protocol change.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from openinterpretability.core.strict import StrictModel

MAX_RIVALS = 3


class RivalReading(StrictModel):
    """One reading of the unit (the hypothesis, or a rival), scored on the measured probes."""

    explanation: str
    balanced_accuracy: float | None = Field(
        default=None, description="On the measured probes that it labels; None with none."
    )
    probes: int = Field(ge=0, description="Measured probes that it labels.")
    disagreements: int = Field(
        ge=0, description="Of those, probes where it predicts otherwise than the hypothesis."
    )


class RivalComparison(StrictModel):
    """Every reading, the hypothesis first, and the one that predicts the probes best."""

    readings: list[RivalReading]
    survivor: int = Field(
        ge=0, description="Index in ``readings`` of the best reading; ties go to the hypothesis."
    )


def balanced_accuracy(predicted: Sequence[bool], actual: Sequence[bool]) -> float | None:
    """The mean recall over the actual classes present, or None with no probes."""
    recalls = []
    for cls in (True, False):
        hits = [p == a for p, a in zip(predicted, actual, strict=True) if a == cls]
        if hits:
            recalls.append(sum(hits) / len(hits))
    return sum(recalls) / len(recalls) if recalls else None


def compare(
    hypothesis: str,
    rivals: Sequence[str],
    predictions: Sequence[tuple[bool, Sequence[bool] | None]],
    actual: Sequence[bool],
) -> RivalComparison | None:
    """Score the hypothesis and each rival on the measured probes, or None without rivals.

    ``predictions`` holds, for each measured probe, what the hypothesis predicts and what
    each rival predicts (or None if the reviewer gave no rival labels). ``actual`` holds
    whether each was active. A probe whose rival labels don't match the rivals in number
    counts for the hypothesis only.
    """
    if not rivals:
        return None
    readings = [_reading(hypothesis, [h for h, _ in predictions], list(actual), None)]
    for i, rival in enumerate(rivals):
        rows = [
            (labels[i], a, h)
            for (h, labels), a in zip(predictions, actual, strict=True)
            if labels is not None and len(labels) == len(rivals)
        ]
        readings.append(
            _reading(rival, [r for r, _, _ in rows], [a for _, a, _ in rows], [h for *_, h in rows])
        )
    best = max(
        range(len(readings)),
        key=lambda i: (readings[i].balanced_accuracy or -1.0, i == 0),
    )
    return RivalComparison(readings=readings, survivor=best)


def _reading(
    explanation: str, predicted: list[bool], actual: list[bool], hypothesis: list[bool] | None
) -> RivalReading:
    score = balanced_accuracy(predicted, actual)
    differs = sum(p != h for p, h in zip(predicted, hypothesis, strict=True)) if hypothesis else 0
    return RivalReading(
        explanation=explanation,
        balanced_accuracy=None if score is None else round(score, 6),
        probes=len(predicted),
        disagreements=differs,
    )


def rival_prompt(rivals: Sequence[str], n: int) -> str:
    """The reviewer's note on the submission's rival readings, or "" without any."""
    from openinterpretability.core.fence import fence

    if not rivals:
        return ""
    listed = "\n".join(f"{i}. {r}" for i, r in enumerate(rivals, 1))
    separating = max(3, n // 5)
    return (
        "Rival readings of the same unit, scored beside the hypothesis on your probes "
        f"(reported, not gated):\n{fence(listed)}\n"
        "For every probe, also give rival_expected_active: what each rival predicts, in this "
        f"order. Write at least {separating} probes where a rival and the hypothesis predict "
        "differently, so the probes can tell the readings apart.\n"
    )
