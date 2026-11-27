"""A greedy plan of unit experiments, for one prompt's coverage or for many at once (#7061).

Coverage doesn't add up: units push the answer both ways, so a big set can bring back less
than a small one. So the plan is greedy. Each pick adds the candidate experiment whose units
raise the planned prompts' coverage most, with the explained units and the earlier picks
kept. With many prompts, a pick's value is summed over the prompts, each capped at the
target. So a unit studied once counts for every prompt that it helps (#5238 §2).

The first-order attribution is a poor guide to a candidate's gain. On pythia-70m, a plan
that tried only the first 120 candidates by attribution stalled near 0.5 coverage. The same
greedy step over every candidate reached 1.0 (#7061). So the plan is *lazy*. It scores every
candidate once, and then rescores only the ``pool`` best by that score at each pick. When no
candidate in the pool helps, it scores every candidate again, and it stops if none helps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from labinacan.progress import Progress

POOL = 256  # candidates rescored at each pick, the best at the last full scan
STEPS = 400  # the most experiments a plan picks
EPSILON = 1e-6  # a pick must raise the plan's value by more than this
# A prompt within this of its target has reached it. The batched measure rounds float32 sums
# in its own order, so a set that brings the whole answer back can read 0.9999.
TOLERANCE = 1e-3


def capped(raw: float | None) -> float:
    """The credited coverage of a raw value: at most 1, and −1 when there is none.

    Args:
        raw: The raw coverage, or None when the prompt has no span to cover.

    Returns:
        The credited coverage.

    Examples:
        >>> capped(1.4), capped(0.3), capped(None)
        (1.0, 0.3, -1.0)
    """
    return -1.0 if raw is None else min(raw, 1.0)


@dataclass
class Plan:
    """The picks of a greedy plan, and each prompt's coverage before and after them."""

    target: float
    now: list[float]
    picked: list[float]
    chosen: list = field(default_factory=list)  # the picked experiments, in pick order
    gains: list[float] = field(default_factory=list)  # each pick's gain, summed over prompts
    scans: int = 0  # how often every candidate was scored

    @property
    def reached(self) -> int:
        """The number of prompts whose coverage reaches the target, within :data:`TOLERANCE`.

        Returns:
            The count.

        Examples:
            >>> Plan(0.9, [0.1, 0.2], [0.8995, 0.5]).reached
            1
        """
        return sum(p >= self.target - TOLERANCE for p in self.picked)


class _Search:
    """The state of one greedy plan: the candidates left and the units kept so far."""

    def __init__(self, scorers: Sequence, candidates: Sequence, target: float):
        """Start with every candidate open and no unit kept.

        Args:
            scorers: One coverage measure for each prompt.
            candidates: The experiments to pick from.
            target: The coverage that each prompt must reach.
        """
        self.scorers, self.target = scorers, target
        self.left, self.units = list(candidates), []
        self.open = {id(e) for e in self.left}  # the candidates not yet picked, for a fast lookup

    def value(self, found: list[float]) -> float:
        """The plan's value: the prompts' coverages, each capped at the target, summed.

        Args:
            found: Each prompt's coverage.

        Returns:
            The value.
        """
        return sum(min(v, self.target) for v in found)

    def score(self, group: Sequence) -> list[list[float]]:
        """Each candidate's coverage on each prompt, with the picks so far kept.

        Args:
            group: The candidates to score.

        Returns:
            For each candidate, its coverage on each prompt.
        """
        if not group:
            return []
        sets = [[*self.units, *e.units] for e in group]
        per_prompt = [[capped(v) for v in s.raw(sets)] for s in self.scorers]
        return [list(col) for col in zip(*per_prompt, strict=True)]

    def rank(self) -> list:
        """Every candidate left, best first by its value now.

        Returns:
            The candidates in order.
        """
        found = self.score(self.left)
        best = sorted(range(len(self.left)), key=lambda i: -self.value(found[i]))
        return [self.left[i] for i in best]

    def best(self, window: list, current: float) -> tuple[int | None, list]:
        """The index in ``window`` of the candidate that raises the value most, and the scores.

        Args:
            window: The candidates to score.
            current: The plan's value before this pick.

        Returns:
            The index, None when no candidate raises the value above ``current``, and each
            candidate's coverages.
        """
        found = self.score(window)
        i = max(range(len(window)), key=lambda k: self.value(found[k]), default=None)
        if i is None or self.value(found[i]) <= current + EPSILON:
            return None, found
        return i, found

    def take(self, pick) -> None:
        """Keep ``pick``'s units for the next picks.

        Args:
            pick: The picked experiment.
        """
        self.left.remove(pick)
        self.open.discard(id(pick))
        self.units += pick.units


def greedy(scorers: Sequence, candidates: Sequence, target: float, *, pool: int = POOL,
           steps: int = STEPS) -> Plan:  # fmt: skip
    """Pick experiments from ``candidates`` until every prompt reaches ``target``.

    Each scorer has ``raw(sets)``: the raw coverage of each set of units on its prompt
    (:class:`openinterpretability.explain_batched.KeptSets`). Each candidate has ``units``.
    The plan stops at the target, when no candidate helps, or after ``steps`` picks.

    Args:
        scorers: One coverage measure for each prompt.
        candidates: The experiments to pick from.
        target: The coverage that each prompt must reach.
        pool: The candidates rescored at each pick.
        steps: The most experiments to pick.

    Returns:
        The plan.

    Examples:
        >>> from types import SimpleNamespace
        >>> prompt = SimpleNamespace(raw=lambda sets: [0.1 + 0.4 * len(s) for s in sets])
        >>> exps = [SimpleNamespace(target=t, units=[t]) for t in "ab"]
        >>> plan = greedy([prompt], exps, 0.8)
        >>> [e.target for e in plan.chosen], plan.reached
        (['a', 'b'], 1)
    """
    search = _Search(scorers, candidates, target)
    now = [capped(s.raw([[]])[0]) for s in scorers]
    plan = Plan(target, now, list(now))
    order: list = []
    if plan.reached < len(scorers):
        plan.scans, order = 1, search.rank()
    progress = Progress("experiments picked for the plan, at most", min(steps, len(search.left)))
    while plan.reached < len(scorers) and search.left and len(plan.chosen) < steps:
        window = [e for e in order if id(e) in search.open][:pool]
        i, found = search.best(window, search.value(plan.picked))
        if i is None:  # the pool's scores are stale: score every candidate again
            plan.scans, order = plan.scans + 1, search.rank()
            window = order[:pool]
            i, found = search.best(window, search.value(plan.picked))
            if i is None:
                break  # no candidate raises the plan's value
        plan.gains.append(sum(found[i]) - sum(plan.picked))
        plan.picked = found[i]
        plan.chosen.append(window[i])
        search.take(window[i])
        progress.step()
    return plan


def candidates_of(plans: Sequence) -> list:
    """The unit experiments of several prompts' plans, each target once, in first-seen order.

    Args:
        plans: Each prompt's planned experiments, as ``explain`` gives them.

    Returns:
        The experiments.

    Examples:
        >>> from types import SimpleNamespace as N
        >>> a, b, other_a = N(target="a"), N(target="b"), N(target="a")
        >>> [e.target for e in candidates_of([N(experiments=[a]), N(experiments=[other_a, b])])]
        ['a', 'b']
    """
    seen: dict = {}
    for planned in plans:
        for e in planned.experiments:
            seen.setdefault(e.target, e)
    return list(seen.values())
