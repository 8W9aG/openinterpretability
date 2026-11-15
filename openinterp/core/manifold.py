"""The multi-dimensional feature check: does a pole and its partners trace a circle or a line?

Engels et al. (2024) find that some features are not one direction but a small subspace.
In GPT-2 layer 7, the days of the week, the months and the years lie on circles. They
found these in clusters of sparse autoencoder features. This check asks the same question
of one neuron pole and its partner poles (``correlated_neurons``). The neuron version is
this registry's step, not theirs.

The harness (:mod:`openinterp.harness.manifold`) measures it. It is reported, not gated
(``docs/PROTOCOL.md`` §4, *What is not evidence (yet)*).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.identifiers import Pole
from openinterp.core.strict import StrictModel, UnitFloat

Family = Literal["weekdays", "months", "numbers"]
Shape = Literal["circle", "line"]
SIGNIFICANT = 0.05  # a family's p-value below this is shown to the proposer as a shape
MIN_ACTIVE = 2  # members the pole must be active on before the proposer sees the family
# A circle's plane must be this round (its smaller variance over its larger) to be shown.
# A flatter plane is a line, and two groups along it in cycle order also score high.
MIN_ROUNDNESS = 0.2


class ManifoldPole(StrictModel):
    """One axis of the check: a neuron pole of the audited layer."""

    neuron_idx: int = Field(ge=0)
    pole: Pole


class FamilyManifold(StrictModel):
    """The joint strengths of the poles over one family of words, and their shape.

    * ``explained_variance``: the share of variance of the first principal components,
      after each pole's strengths are scaled to unit variance.
    * ``components``: the 1-based components that ``score`` uses. A circle uses a plane,
      the pair 1–2 or 2–3. A line uses one component.
    * ``score``: for a circle, the mean resultant length of each member's angle in that
      plane minus its place in the cycle. The better of the two directions counts. 1 is
      a circle in order. For a line, Spearman's |rho| between the component and the value.
    * ``p_value``: the share of shuffled orders that score at least as high, over the same
      choice of components.
    * ``pole_share``: the share of the audited pole's axis that lies in those components.
      Near 0, the shape belongs to the partners, not to the pole.
    * ``roundness``: for a circle, the plane's smaller variance over its larger. Near 0,
      the plane is a line, so weekend against weekday can score as a circle.
    """

    family: Family
    shape: Shape
    members: int = Field(ge=0)
    active_members: int = Field(ge=0)
    explained_variance: list[UnitFloat]
    components: list[int]
    score: UnitFloat
    p_value: UnitFloat
    pole_share: UnitFloat
    roundness: UnitFloat | None = None


class ManifoldCheck(StrictModel):
    """The check for one neuron pole: its axes, the activating set, and each family.

    ``poles`` lists the audited pole first, then its partners. The activating set is the
    peak token of each top example. Its ``participation_ratio`` is the effective number of
    dimensions, from 1 (one direction) to the number of poles.
    """

    poles: list[ManifoldPole] = Field(min_length=1)
    activating_examples: int = Field(ge=0)
    activating_explained_variance: list[UnitFloat]
    participation_ratio: float = Field(ge=0.0)
    families: list[FamilyManifold]


def shown_families(check: ManifoldCheck) -> list[FamilyManifold]:
    """The families with a shape (p below :data:`SIGNIFICANT`) on which the pole is active.

    A circle must also be round (:data:`MIN_ROUNDNESS`).
    """
    return [
        f
        for f in check.families
        if f.p_value < SIGNIFICANT
        and f.active_members >= MIN_ACTIVE
        and (f.shape != "circle" or (f.roundness or 0.0) >= MIN_ROUNDNESS)
    ]


def format_for_proposer(check: ManifoldCheck) -> str:
    """The proposer's note on the check, with the allowed hypothesis form when a shape shows."""
    axes = ", ".join(f"N{p.neuron_idx}{'+' if p.pole == 'positive' else '-'}" for p in check.poles)
    note = (
        f"\nJoint strengths of this pole and its partners ({axes}) on the activating set: "
        f"{check.participation_ratio:.1f} effective dimensions of {len(check.poles)}.\n"
    )
    for f in shown_families(check):
        plane = "–".join(f"PC{c}" for c in f.components)
        note += (
            f"On the {f.family} ({f.active_members} of {f.members} active), the poles trace a "
            f"{f.shape} in {plane} (score {f.score:.2f}, p = {f.p_value:.3f}, "
            f"{f.pole_share:.0%} of this pole in it).\n"
        )
    if shown_families(check):
        note += (
            "The pole may be one coordinate of a multi-dimensional feature. You may say so, "
            "but the explanation must still predict this pole's strength on each text, for "
            "example which members it fires on most and least.\n"
        )
    return note
