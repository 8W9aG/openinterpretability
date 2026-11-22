"""Draws that run one answer's planned experiments until each unit is VERIFIED (issue #3744).

``open-interpretability explain --run`` takes the experiments of its plan, units and
superpositions together, by coverage gain (:mod:`openinterpretability.explain_plan`). It runs
until the explained share reaches the target (:class:`ShareMeter`, #4473), or no target is
left. ``--run N`` stops after N experiments too. It runs them through the
experiment loop (:class:`openinterpretability.experiment_loop.Loop`), with
:class:`TargetedDraws` in place of the planner's draw. So each run is a normal experiment:
claimed, audited, reviewed, verified, opened as a PR and landed.

Each draw reloads the registry and takes the first target that still needs work:

- an unstudied unit gets an audit chosen by attribution, as the plan gives it;
- an unstudied superposition gets the planner's superposition draw;
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

    slug: str  # NeuronIdentifier, HeadIdentifier or SuperpositionIdentifier slug
    model: str
    layer: int
    audit: str  # the full `open-interpretability audit …` command; empty for a superposition
    label: str  # L3/N7+, L2/H4 or L20/S0123456789ab, for the summary
    definition: dict | None = None  # a superposition's members, for its first draw


def by_gain(planned) -> list:
    """Planned experiments by the prompt coverage each would add, highest first.

    Those with no measured gain keep their plan order, after the others.
    """
    return sorted(planned, key=lambda e: (e.gain is None, -(e.gain or 0.0)))


def targets(planned, model: str, groups=()) -> list[Target]:
    """The plan's unit and superposition experiments, in their order, as targets.

    ``groups`` are the model's :class:`~openinterpretability.explain_groups.GroupDef`, for each
    superposition's definition. A circuit is not a target (:func:`circuit_notes`).
    """
    from openinterpretability.explain import _slug
    from openinterpretability.explain_plan import audit_command

    definitions = {g.name: g for g in groups if g.kind == "superposition"}
    out = []
    for e in planned:
        if e.entity == "superposition" and e.target in definitions:
            g = definitions[e.target]
            layer = int(e.target.split("/")[0][1:])
            out.append(Target(g.slug, model, layer, "", e.target, g.definition))
        elif e.entity in ("pole", "head"):
            (unit,) = e.units
            audit = next((c for c in e.commands if c.startswith("open-interpretability audit ")),
                         audit_command(model, unit, None))  # fmt: skip
            out.append(Target(_slug(model, unit), model, unit[0], audit, e.target))
    return out


def circuit_notes(planned) -> list[str]:
    """Why each planned circuit experiment can't run in the loop, with what would run it.

    A circuit's task needs a corrupted twin of the prompt: one of the same token length
    that predicts another token. Choosing it is a judgement about which part of the prompt
    carries the answer. The API loop has no agent that writes tasks (``AGENT_ONLY_KINDS``).
    """
    out = []
    for e in planned:
        if e.entity == "circuit":
            why = f"{e.target} ({e.share:.1%}): its task needs a corrupted twin of this prompt"
            out.append(f"{why}, which an agent writes. Run: " + " && ".join(e.commands))
    return out


def explained_now(model: str, order: list, groups: list, wiki_dir: Path) -> set:
    """The units of ``order`` that are explained in the registry now.

    A unit is explained when it is VERIFIED, or when a VERIFIED superposition or circuit
    holds it, as :func:`openinterpretability.explain.explain` counts it.
    """
    from openinterpretability.explain import _slug, registry_labels
    from openinterpretability.explain_groups import circuit_labels

    labels = registry_labels(wiki_dir, model)
    circuits = circuit_labels(wiki_dir, model)
    covered = {u for g in groups if _group_status(g, labels, circuits) == "VERIFIED"
               for u in g.units}  # fmt: skip
    verified = {u for u, *_ in order if labels.get(_slug(model, u), ("",))[0] == "VERIFIED"}
    return covered | verified


@dataclass
class ShareMeter:
    """The prompt's explained share of |attribution|, measured again from the registry.

    The prompt's attributions don't change, only the registry's statuses.
    """

    model: str
    order: list  # explain's (unit key, |attribution|, status, explained), in rank order
    groups: list  # the GroupDefs that carry the answer
    wiki_dir: Path
    note: str = ""

    def __call__(self) -> float:
        """The share of the total |attribution| that is explained now."""
        done = explained_now(self.model, self.order, self.groups, self.wiki_dir)
        total = sum(a for _, a, _, _ in self.order) or 1.0
        return sum(a for unit, a, _, _ in self.order if unit in done) / total


@dataclass
class CoverageMeter:
    """The prompt's causal coverage, measured again after each landed experiment (#4476).

    The explained units are kept, the rest mean-ablated, and the share of the answer's
    log-probability that comes back is the coverage. Random sets of the same count are the
    baseline. Coverage is noisy for small sets, so it counts only when it beats every random
    set. Otherwise the meter gives 0. :attr:`note` says both numbers.
    """

    runner: object
    means: object
    prompt: str
    token: int
    position: int
    model: str
    order: list
    groups: list
    wiki_dir: Path
    draws: int = 4
    note: str = ""

    def __call__(self) -> float:
        """The coverage, when it beats every random set, else 0."""
        from openinterpretability.explain_coverage import prompt_coverage

        done = explained_now(self.model, self.order, self.groups, self.wiki_dir)
        order = [(u, a, s, u in done) for u, a, s, _ in self.order]
        found = prompt_coverage(self.runner, self.prompt, self.means, self.token, self.position,
                                order, draws=self.draws)  # fmt: skip
        coverage = found["coverage"] if found["coverage"] is not None else 0.0
        best = max(found["random_draws"], default=0.0)
        self.note = f" (coverage {coverage:.1%}, best random set {best:.1%})"
        return coverage if coverage > best else 0.0


def _group_status(g, labels: dict, circuits: dict) -> str:
    if g.kind == "superposition":
        return labels.get(g.slug, ("unstudied",))[0]
    return circuits.get(g.slug, ("unstudied",))[0]


@dataclass
class TargetedDraws:
    """A ``plan_next`` for the loop that draws only for ``targets`` (module docstring)."""

    targets: list[Target]
    reviewer: str
    max_attempts: int = MAX_ATTEMPTS
    attempts: dict[str, int] = field(default_factory=dict)
    done: dict[str, str] = field(default_factory=dict)  # slug → why it left the list
    meter: ShareMeter | None = None  # with it, the run stops at ``target``
    target: float = 1.0
    limit: int = 0  # experiments to start at most; 0 for no limit
    window: int | None = None  # the model's scan window, for a superposition's draw
    started: int = 0
    shares: list = field(default_factory=list)  # each measured share, for the page (#4479)

    def __call__(self, wiki_dir: Path, *, epsilon: float = 0.0, claims=()):  # noqa: ARG002
        # The keywords of planner.plan_next, so the loop calls either one the same way.
        """The next experiment for the first target that needs one, as a planner ``Plan``."""
        from openinterpretability.planner import Plan, claim_key, load_states, stopped

        self._stop_if_done()
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
            self.started += 1
            self._say_start(t)
            return Plan(seed=0, choice=choice, explored=False, candidates=[choice])
        raise Finished(self.summary())

    def _say_start(self, t: Target) -> None:
        """Say which experiment starts: its count of the run's, its unit and its attempt (#5830).

        Args:
            t: The target that the experiment is for.
        """
        from openinterpretability.experiment_loop import say

        of = self.limit or len(self.targets)
        say(f"experiment {self.started}/{of}: {t.label}, attempt {self.attempts[t.slug]}")

    def _stop_if_done(self) -> None:
        """Say the explained share, and end the run at the target or the limit (#4473).

        A draw comes after the previous experiment landed, so the share is measured here.
        """
        from openinterpretability.experiment_loop import say

        if self.meter is not None:
            share = self.meter()
            self.shares.append(round(share, 4))
            note = getattr(self.meter, "note", "")
            say(f"measured {share:.1%}{note}, target {self.target:.0%}")
            if share >= self.target:
                raise Finished(f"the measure {share:.1%} reached the target "
                               f"{self.target:.0%}. {self.summary()}")  # fmt: skip
        if self.limit and self.started >= self.limit:
            raise Finished(f"ran {self.started} experiments. {self.summary()}")

    def _finished(self, t: Target, state, stops: set[str]) -> str | None:
        """Why a target needs no more runs, or None."""
        if state is not None and state.standing[0] == "VERIFIED":  # the standing (#4900)
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

        if state is None and t.definition is not None:
            from openinterpretability.planner_superpositions import (
                _candidate,
                default_definitions,
                definition_identifier,
            )

            ident = definition_identifier(t.definition)
            return _candidate(ident, t.definition, default_definitions(t.model), self.window)
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
