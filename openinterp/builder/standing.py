"""A unit's standing: what all of its experiments say now (``docs/PROTOCOL.md`` §3 step 9).

The newest hypothesis counts, one reading per probe set, independent reviews outweigh
self-reviews, and the unit is CONTESTED when those readings disagree on VERIFIED. Also the
registry-wide reviewer agreement those readings give. Derived, never stored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openinterp.core.agent_names import same_agent

if TYPE_CHECKING:
    from openinterp.builder.tree import Experiment, ModelNode

# Why an experiment does or doesn't count towards its pole's standing, with its label.
ROLES = {
    "current": "counts toward status",
    "remeasured": "re-measured since",
    "outweighed": "outweighed by an independent review",
    "earlier_hypothesis": "earlier hypothesis",
}


@dataclass
class Standing:
    """What a pole's experiments say now. Derived, never stored (``pole_standing``)."""

    status: str  # VERIFIED, NEEDS_REVISION, REJECTED, or CONTESTED when readings disagree
    readings: list[Experiment]  # the experiments it rests on, newest first

    @property
    def lead(self) -> Experiment:
        return self.readings[0]

    @property
    def verified(self) -> bool:
        return self.status == "VERIFIED"

    @property
    def contested(self) -> bool:
        return self.status == "CONTESTED"

    @property
    def self_reviewed(self) -> bool:
        """Only self-reviews back it: an independent review would have outweighed them."""
        return all(e.record.self_reviewed for e in self.readings)


def _newest_first(experiments: Sequence[Experiment]) -> list[Experiment]:
    return sorted(experiments, key=lambda e: (e.record.verdict.timestamp, e.id), reverse=True)


def _readings(experiments: Sequence[Experiment]) -> list[Experiment]:
    """The newest experiment per probe set, newest first: a re-measurement (e.g. under a
    newer protocol) replaces the reading it repeats instead of counting twice."""
    seen: set[str] = set()
    out = []
    for e in _newest_first(experiments):
        if e.probe_set not in seen:
            seen.add(e.probe_set)
            out.append(e)
    return out


def pole_standing(experiments: Sequence[Experiment]) -> Standing:
    """Derive one pole's standing from all of its experiments, and set each one's ``role``.

    1. Only the newest hypothesis counts: a new submission replaces older ones, whose
       verdicts were about a different claim.
    2. Each probe set counts once, read by its newest experiment.
    3. If any reading is an independent review, self-reviews are outweighed.
    4. The readings left agree, or the pole is CONTESTED. Only VERIFIED against anything
       else is a disagreement; if none is VERIFIED, the newest reading's status stands.
    """
    newest = _newest_first(experiments)
    introduced = {e.claim: e for e in reversed(newest)}  # each claim's first experiment
    current = max(introduced.values(), key=lambda e: (e.record.verdict.timestamp, e.id)).claim
    latest = _readings([e for e in newest if e.claim == current])
    readings = [e for e in latest if not e.record.self_reviewed] or latest
    counted, read = {e.id for e in readings}, {e.id for e in latest}
    for e in newest:
        if e.claim != current:
            e.role = "earlier_hypothesis"
        elif e.id in counted:
            e.role = "current"
        elif e.id in read:
            e.role = "outweighed"
        else:
            e.role = "remeasured"
    statuses = {e.record.verdict.status for e in readings}
    contested = "VERIFIED" in statuses and len(statuses) > 1
    standing = Standing("CONTESTED" if contested else readings[0].record.verdict.status, readings)
    for e in newest:
        e.standing = standing
    return standing


def native_standings(models: Sequence[ModelNode]):
    """``(unit, side, standing)`` for every native neuron pole and head (dictionary features
    are a baseline, not claims about the model's own units)."""
    for m in models:
        for layer in m.layers:
            for n in layer.units:
                if not n.is_feature:
                    for side, standing in n.standings.items():
                        yield n, side, standing


def reviewer_agreement(models: Sequence[ModelNode]) -> dict:
    """How often reviews of the same hypothesis agree on VERIFIED versus not.

    Counts every hypothesis about a native unit with more than one probe set (re-reviews,
    replications, independent re-reviews), each probe set read by its newest experiment,
    self-reviews included: this measures the reviews, not the standing rule.
    """
    claims = unanimous = pairs = agreeing = 0
    cross_pairs = cross_agreeing = 0
    for m in models:
        for layer in m.layers:
            for n in layer.units:
                if n.is_feature:
                    continue
                by_claim: dict[tuple[str, str], list[Experiment]] = {}
                for e in n.experiments:
                    by_claim.setdefault((e.side, e.claim), []).append(e)
                for group in by_claim.values():
                    readings = _readings(group)
                    if len(readings) < 2:
                        continue
                    claims += 1
                    unanimous += len({e.record.is_verified for e in readings}) == 1
                    for i, a in enumerate(readings):
                        for b in readings[i + 1 :]:
                            agree = a.record.is_verified == b.record.is_verified
                            pairs += 1
                            agreeing += agree
                            ra, rb = a.record.verdict, b.record.verdict
                            if not same_agent(ra.reviewer_agent_model, rb.reviewer_agent_model):
                                cross_pairs += 1
                                cross_agreeing += agree
    return {
        "claims": claims,
        "unanimous": unanimous,
        "pairs": pairs,
        "pair_agreement": round(agreeing / pairs, 4) if pairs else None,
        "cross_model_pairs": cross_pairs,
        "cross_model_agreement": round(cross_agreeing / cross_pairs, 4) if cross_pairs else None,
    }
