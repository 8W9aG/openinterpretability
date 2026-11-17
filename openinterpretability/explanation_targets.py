"""Draws that run one answer's planned experiments until each unit is VERIFIED (issue #3744).

``open-interpretability explain --run N`` takes the top N unit experiments of its plan, by
coverage gain (:mod:`openinterpretability.explain_plan`). It runs them through the
experiment loop (:class:`openinterpretability.experiment_loop.Loop`), with
:class:`TargetedDraws` in place of the planner's draw. So each run is a normal experiment:
claimed, audited, reviewed, verified, opened as a PR and landed.

Each draw reloads the registry and takes the first target that still needs work:

- an unstudied unit gets an audit chosen by attribution, as the plan gives it;
- a studied unit that isn't VERIFIED gets the planner's heaviest follow-up on its record
  (``planner.candidates_for``). A review of the reviewer's own record is passed over.

A unit stays a target until it stands VERIFIED. So one unit can take several runs:
revise, a new hypothesis, an independent review and so on. A unit is dropped when it is
VERIFIED, when a scan stops on it (dead or default-on), or when it has used
``max_attempts`` runs. One that an open PR claims waits. When no target is left,
:class:`~openinterpretability.experiment_loop.Finished` ends the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from openinterpretability.experiment_loop import Finished

MAX_ATTEMPTS = 5  # runs per unit before the loop gives up on it


@dataclass(frozen=True)
class Target:
    """One unit to explain, and the audit that a first experiment on it runs."""

    slug: str  # NeuronIdentifier or HeadIdentifier slug
    model: str
    layer: int
    audit: str  # the full `open-interpretability audit …` command
    label: str  # L3/N7+ or L2/H4, for the summary


def targets(planned, model: str) -> list[Target]:
    """The plan's unit experiments, in its order (coverage gain first), as targets."""
    from openinterpretability.explain import _slug
    from openinterpretability.explain_plan import audit_command

    out = []
    for e in planned:
        (unit,) = e.units
        audit = next((c for c in e.commands if c.startswith("open-interpretability audit ")),
                     audit_command(model, unit, None))  # fmt: skip
        out.append(Target(_slug(model, unit), model, unit[0], audit, e.target))
    return out


@dataclass
class TargetedDraws:
    """A ``plan_next`` for the loop that draws only for ``targets`` (module docstring)."""

    targets: list[Target]
    reviewer: str
    max_attempts: int = MAX_ATTEMPTS
    attempts: dict[str, int] = field(default_factory=dict)
    done: dict[str, str] = field(default_factory=dict)  # slug → why it left the list

    def __call__(self, wiki_dir: Path, *, epsilon: float = 0.0, claims=()):
        """The next experiment for the first target that needs one, as a planner ``Plan``."""
        from openinterpretability.planner import Plan, claim_key, load_states, stopped

        states, now = load_states(wiki_dir), datetime.now(UTC)
        stops = {ident.slug for ident in stopped(wiki_dir)}
        claimed = {claim for claim in claims}
        for t in self.targets:
            if t.slug in self.done:
                continue
            state = states.get(t.slug)
            if why := self._finished(t, state, stops):
                self.done[t.slug] = why
                continue
            if any(claim_key(t.slug) in c for c in claimed):
                continue  # an open PR studies it now: wait for it to land
            choice = self._choice(t, state, set(states), claims, now)
            if choice is None:
                self.done[t.slug] = "no follow-up the reviewer can run"
                continue
            self.attempts[t.slug] = self.attempts.get(t.slug, 0) + 1
            return Plan(seed=0, choice=choice, explored=False, candidates=[choice])
        raise Finished(self.summary())

    def _finished(self, t: Target, state, stops: set[str]) -> str | None:
        """Why a target needs no more runs, or None."""
        if state is not None and state.record.verdict.status == "VERIFIED":
            return "VERIFIED"
        if t.slug in stops:
            return "stopped (dead or default-on)"
        if self.attempts.get(t.slug, 0) >= self.max_attempts:
            return f"not VERIFIED after {self.max_attempts} runs"
        return None

    def _choice(self, t: Target, state, slugs: set[str], claims, now):
        """An audit for an unstudied unit, else its heaviest follow-up that isn't self-review."""
        from openinterpretability.draw_steps import REVIEW_KINDS
        from openinterpretability.experiment_loop import proposer_of
        from openinterpretability.planner import SKILLS, Candidate, _audit_options, candidates_for

        if state is None:
            why = "a unit that the answer chose, unstudied: its audit records selection attribution"
            return Candidate(kind="explore", model=t.model, weight=0.0, rationale=why,
                             commands=[t.audit], slug=t.slug, skills=SKILLS["explore"],
                             layer=t.layer, audit=_audit_options([t.audit]))  # fmt: skip
        found = sorted(candidates_for(state, slugs, claims, now), key=lambda c: -c.weight)
        for c in found:
            if c.kind in REVIEW_KINDS and proposer_of(c.record) == self.reviewer:
                continue  # the reviewer can't review its own record independently
            return c
        return None

    def summary(self) -> str:
        """Each target's outcome, for the loop's last line."""
        parts = [f"{t.label} {self.done.get(t.slug, 'waiting on an open PR')}"
                 for t in self.targets]  # fmt: skip
        return "; ".join(parts)
