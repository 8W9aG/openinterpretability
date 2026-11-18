"""A unit's standing: what all of its experiments say now (``docs/PROTOCOL.md`` §3 step 9).

The newest hypothesis counts, one reading per probe set, independent reviews outweigh
self-reviews, and a majority of those readings decides. Only VERIFIED against REJECTED is a
disagreement: NEEDS_REVISION never contests a VERIFIED claim (issue #4900). An even split
between VERIFIED and REJECTED readings goes to the side with the higher detection score.
When the scores are close, the unit is CONTESTED (#3212). A newer REJECTED hypothesis after a
VERIFIED one is settled by the scores the same way (#4654). An erratum
can set experiments aside (:mod:`openinterpretability.core.errata`). Also the
registry-wide reviewer agreement those readings give. Derived, never stored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openinterpretability.core.agent_names import same_agent

if TYPE_CHECKING:
    from openinterpretability.builder.tree import Experiment, ModelNode

# Why an experiment does or doesn't count towards its pole's standing, with its label.
ROLES = {
    "current": "counts toward status",
    "remeasured": "re-measured since",
    "outweighed": "outweighed by an independent review",
    "earlier_hypothesis": "earlier hypothesis",
    "contested_hypothesis": "earlier VERIFIED hypothesis, contested by a newer one",
    "standing_hypothesis": "VERIFIED hypothesis that a newer, failed one doesn't overturn",
    "failed_rival": "newer hypothesis that failed, and doesn't overturn the VERIFIED one",
    "set_aside": "set aside by an erratum",
}


@dataclass
class Standing:
    """What a pole's experiments say now. Derived, never stored (``pole_standing``)."""

    status: str  # VERIFIED, NEEDS_REVISION, REJECTED or CONTESTED
    readings: list[Experiment]  # the experiments it rests on: those that agree first, newest first
    # The lead of an earlier VERIFIED hypothesis that a newer, failed one contests (#4654),
    # or that keeps the standing against it (#4900). None when no newer hypothesis failed.
    prior: Experiment | None = None

    @property
    def margin(self) -> float:
        """The lead in score that settles VERIFIED against REJECTED, for the pages (#4900)."""
        return SCORE_MARGIN

    @property
    def lead(self) -> Experiment:
        """The newest experiment that the standing rests on."""
        return self.readings[0]

    @property
    def verified(self) -> bool:
        """True when the standing is VERIFIED."""
        return self.status == "VERIFIED"

    @property
    def broad(self) -> bool:
        """True when the standing is VERIFIED and its unit is broad (#4545)."""
        from openinterpretability.core.explanation import is_broad

        return self.verified and is_broad(self.lead.record.verdict)

    @property
    def contested(self) -> bool:
        """True when the standing is CONTESTED, by an even split or by a newer hypothesis."""
        return self.status == "CONTESTED"

    @property
    def contested_by_hypothesis(self) -> bool:
        """True when a newer hypothesis failed after an earlier VERIFIED one (#4654)."""
        return self.contested and self.prior is not None

    @property
    def agreeing(self) -> list[Experiment]:
        """The readings whose verdict is on the side of the standing: VERIFIED or not."""
        verified = self.status == "VERIFIED"
        return [e for e in self.readings if (e.record.verdict.status == "VERIFIED") == verified]

    @property
    def dissent(self) -> int:
        """How many readings the majority outvoted; 0 when all agree or the unit is contested."""
        return 0 if self.contested else len(self.readings) - len(self.agreeing)

    @property
    def self_reviewed(self) -> bool:
        """Only self-reviews back it: an independent review would have outweighed them."""
        return all(e.record.self_reviewed for e in self.readings)

    @property
    def errata(self) -> list:
        """The errata that cover a reading it rests on, each once, in first-seen order."""
        return list({x.id: x for e in self.readings for x in e.errata}.values())


def _newest_first(experiments: Sequence[Experiment]) -> list[Experiment]:
    return sorted(experiments, key=lambda e: (e.record.verdict.timestamp, e.id), reverse=True)


def _readings(experiments: Sequence[Experiment]) -> list[Experiment]:
    """Return the newest experiment per probe set, newest first.

    A re-measurement (for example, under a newer protocol) replaces the reading that it
    repeats, so it does not count twice.
    """
    seen: set[str] = set()
    out = []
    for e in _newest_first(experiments):
        if e.probe_set not in seen:
            seen.add(e.probe_set)
            out.append(e)
    return out


SCORE_MARGIN = 0.05  # a lead in detection score that settles VERIFIED against REJECTED (#4900)


def score(e: Experiment) -> float | None:
    """The reading's detection score on held-out windows, else its probe balanced accuracy.

    The maintainer chose the held-out score on 2026-11-18 (#4900): neither agent chose its
    text, so it compares two hypotheses fairly.
    """
    verdict = e.record.verdict
    scores = verdict.explanation_scores
    if scores is not None and scores.detection is not None:
        return scores.detection.balanced_accuracy
    return verdict.metrics.balanced_accuracy if verdict.metrics is not None else None


def by_score(verified: Experiment, rejected: Experiment) -> str:
    """The side that scores higher by :data:`SCORE_MARGIN`; CONTESTED when neither does."""
    a, b = score(verified), score(rejected)
    if a is None or b is None or abs(a - b) < SCORE_MARGIN:
        return "CONTESTED"
    return "VERIFIED" if a > b else "REJECTED"


def _majority(readings: list[Experiment]) -> Standing:
    """The standing that ``readings`` (newest first) give (#3212, #4900).

    NEEDS_REVISION never contests VERIFIED. Between VERIFIED and REJECTED readings the
    majority decides, and an even split goes to :func:`by_score`.
    """

    def of(status: str) -> list[Experiment]:
        return [e for e in readings if e.record.verdict.status == status]

    verified, rejected = of("VERIFIED"), of("REJECTED")
    rest = [e for e in readings if e not in verified]
    if not verified:
        return Standing(readings[0].record.verdict.status, readings)
    if len(verified) > len(rejected):
        return Standing("VERIFIED", verified + rest)
    if len(rejected) > len(verified):
        return Standing("REJECTED", rejected + [e for e in readings if e not in rejected])
    status = by_score(verified[0], rejected[0])
    if status == "REJECTED":
        return Standing(status, rejected + [e for e in readings if e not in rejected])
    return Standing(status, verified + rest if status == "VERIFIED" else readings)


def _claim_readings(newest: list[Experiment], claim: str) -> tuple[list, list]:
    """Return ``(latest, counted)`` for one claim: steps 2 and 3 of :func:`pole_standing`.

    ``latest`` is the newest reading per probe set. ``counted`` is those left after
    independent reviews outweigh self-reviews.
    """
    latest = _readings([e for e in newest if e.claim == claim])
    return latest, [e for e in latest if not e.record.self_reviewed] or latest


def _verified_prior(newest: list[Experiment], current: str, introduced: dict) -> list | None:
    """Return the counted readings of the newest earlier claim that is VERIFIED, or None.

    Each earlier claim is scored by steps 2–4 on its own experiments. That is the status
    that it had while it was the newest, because a later claim never adds readings to it.
    """
    earlier = sorted(
        (e for claim, e in introduced.items() if claim != current),
        key=lambda e: (e.record.verdict.timestamp, e.id),
        reverse=True,
    )
    for first in earlier:
        _, counted = _claim_readings(newest, first.claim)
        if _majority(counted).verified:
            return counted
    return None


def _key(e: Experiment) -> tuple:
    return e.record.verdict.timestamp, e.id


def _against_prior(standing: Standing, prior: list[Experiment], newest: list) -> Standing:
    """Step 5 of :func:`pole_standing`: failed newer hypotheses against a VERIFIED one.

    Only a REJECTED reading of a newer hypothesis disagrees with it. The newest such reading
    goes to :func:`by_score`, whichever hypothesis is newest now.
    """
    since = min(_key(e) for e in prior)
    rejected = [e for e in newest if e.claim != prior[0].claim and _key(e) > since
                and e.record.verdict.status == "REJECTED"]  # fmt: skip
    status = by_score(prior[0], rejected[0]) if rejected else "VERIFIED"
    if status == "REJECTED":
        return standing  # the newer hypothesis outscores it: its own status stands
    role = "standing_hypothesis" if status == "VERIFIED" else "contested_hypothesis"
    for e in prior:
        e.role = role
    if status == "VERIFIED":  # it rests on the earlier hypothesis's readings (#4900)
        for e in standing.readings:
            e.role = "failed_rival" if e.role == "current" else e.role
        return Standing("VERIFIED", prior, prior=prior[0])
    return Standing("CONTESTED", standing.readings, prior=prior[0])


def pole_standing(experiments: Sequence[Experiment]) -> Standing:
    """Derive one pole's standing from all of its experiments, and set each one's ``role``.

    1. Only the newest hypothesis counts: a new submission replaces older ones, whose
       verdicts were about a different claim.
    2. Each probe set counts once, read by its newest experiment.
    3. If any reading is an independent review, self-reviews are outweighed.
    4. The readings left decide (:func:`_majority`). NEEDS_REVISION never counts against
       VERIFIED. Between VERIFIED and REJECTED readings, the majority decides. An even
       split goes to :func:`by_score`, and is CONTESTED when the scores are close (#3212,
       #4900).
    5. Suppose step 4 gives a status other than VERIFIED, and an earlier hypothesis was
       VERIFIED by steps 2–4. Then a failed rival does not simply overturn it (#4654,
       #4900). A newer NEEDS_REVISION never contests it: the earlier hypothesis keeps the
       standing. A newer REJECTED goes to :func:`by_score`. The result is VERIFIED on the
       earlier hypothesis, REJECTED on the newer one, or CONTESTED when the scores are close.
       A newer hypothesis that reaches VERIFIED settles it.

    Before step 1, experiments that an erratum sets aside are left out, unless that leaves
    none. Then the standing rests on them, and :attr:`Standing.errata` names the errata.
    """
    kept = [e for e in experiments if not e.set_aside]
    for e in experiments:
        if kept and e.set_aside:
            e.role = "set_aside"
    newest = _newest_first(kept or experiments)
    introduced = {e.claim: e for e in reversed(newest)}  # each claim's first experiment
    current = max(introduced.values(), key=lambda e: (e.record.verdict.timestamp, e.id)).claim
    latest, readings = _claim_readings(newest, current)
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
    standing = _majority(readings)
    prior = None if standing.verified else _verified_prior(newest, current, introduced)
    if prior:
        standing = _against_prior(standing, prior, newest)
    for e in experiments:
        e.standing = standing
    return standing


def native_standings(models: Sequence[ModelNode]):
    """Return ``(unit, side, standing)`` for every native neuron pole and head.

    Dictionary features and superpositions are left out. They are not claims about one
    of the model's own units.
    """
    for m in models:
        for layer in m.layers:
            for n in layer.units:
                if n.is_native:
                    for side, standing in n.standings.items():
                        yield n, side, standing


def contested_units(models: Sequence[ModelNode]) -> list[dict]:
    """Every CONTESTED native pole or head, as rows for the index page."""
    return [
        {"model": n.model, "unit": n, "side": side, "standing": st}
        for n, side, st in native_standings(models)
        if st.contested
    ]


def reviewer_agreement(models: Sequence[ModelNode]) -> dict:
    """How often reviews of the same hypothesis agree on VERIFIED versus not.

    Counts every hypothesis about a native unit with more than one probe set (re-reviews,
    replications, independent re-reviews). Each probe set is read by its newest
    experiment. Self-reviews are included, because this measures the reviews, not the
    standing rule.
    """
    claims = unanimous = pairs = agreeing = 0
    cross_pairs = cross_agreeing = 0
    for m in models:
        for layer in m.layers:
            for n in layer.units:
                if not n.is_native:
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
