"""The experiments that a complete explanation of one answer needs (issue #3742).

``open-interpretability explain`` ranks every unit by |attribution| and lists the
superpositions and circuits that carry the answer (:mod:`openinterpretability.explain`).
:func:`plan` lists an experiment for each of them that is not VERIFIED:

- *Poles and heads*, in rank order, until the explained share plus the planned share
  reaches ``target``. An unstudied unit gets an audit with ``--selection attribution``,
  because the answer chose it, so its result stays out of the random-selection
  statistics. A studied unit gets the planner's heaviest follow-up on its own record
  (``planner.candidates_for``). A stopped unit (dead or default-on), or a studied unit with
  no follow-up, can't be planned, and its share goes to ``blocked``.
- *Superpositions*: an unstudied one gets the planner's superposition draw, and a
  studied one its heaviest follow-up.
- *Circuits*: a task hypothesis on this prompt (``circuits propose`` and ``circuits
  review``). One prompt has no corrupted twin, so the task file must be written first.

A share is of this prompt's first-order attribution. A VERIFIED explanation describes its
unit on the scan corpus, so it may not cover this prompt.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openinterpretability.explain_groups import HEAD

TARGET = 1.0  # a complete explanation; --target lowers it
CAVEAT = (
    "Shares are of this prompt's first-order attribution. A VERIFIED explanation describes "
    "its unit on the scan corpus, so it may not cover this prompt."
)


@dataclass
class PlannedExperiment:
    """One experiment that would raise the explained share."""

    target: str  # L3/N7+, L3/H2, L20/S0123456789ab or C4
    entity: str  # pole, head, superposition or circuit
    share: float  # of the total |attribution|; a group's overlaps its members'
    status: str
    draw: str  # explore, revise, new_hypothesis, superposition, circuit, …
    rationale: str
    commands: list[str]
    covered_by: list[str] = field(default_factory=list)  # planned groups that hold the unit
    units: list = field(default_factory=list, repr=False)  # the unit keys it would explain
    gain: float | None = None  # the prompt coverage it would add (explain_coverage.add_gains)


@dataclass
class Plan:
    """The experiments, and what they reach."""

    target: float
    explained: float  # the share explained now
    planned_share: float  # the share the planned unit experiments would add
    blocked: float  # the share that no experiment can raise
    experiments: list[PlannedExperiment]  # poles and heads, in rank order
    groups: list[PlannedExperiment]  # superpositions and circuits, by |attribution|
    blocked_units: list[dict]  # {target, share, reason}

    @property
    def reaches(self) -> float:
        """The explained share once every planned unit experiment is VERIFIED."""
        return self.explained + self.planned_share


def audit_command(model: str, unit: tuple[int, int, str], window: int | None) -> str:
    """The audit of an unstudied pole or head, chosen by attribution."""
    layer, index, pole = unit
    which = f"--head {int(index)}" if pole == HEAD else f"--neuron {int(index)} --pole {pole}"
    if pole not in (HEAD, "positive", "negative"):
        raise ValueError(f"not a pole: {pole!r}")
    command = (
        f"open-interpretability audit --model {shlex.quote(model)} --layer {int(layer)} {which} "
        "--selection attribution"
    )
    return command + (f" --window-tokens {window}" if window else "")


def _follow_up(state, slugs: set[str], now: datetime):
    """The planner's heaviest follow-up on a studied unit, or None."""
    from openinterpretability.planner import candidates_for

    found = list(candidates_for(state, slugs, (), now))
    return max(found, key=lambda c: c.weight) if found else None


def _unit(model, unit, status, states, window, slugs, now):
    """``(draw, rationale, commands)`` for one unexplained unit, or the reason it is blocked."""
    from openinterpretability.explain import _slug

    if status == "stopped":
        return "a scan stopped on it (dead or default-on)"
    if status == "unstudied":
        why = "unstudied; the answer chose it, so the audit records selection attribution"
        return "explore", why, [audit_command(model, unit, window)]
    state = states.get(_slug(model, unit))
    best = _follow_up(state, slugs, now) if state is not None else None
    if best is None:
        return f"{status}, and the planner has no follow-up for it"
    return best.kind, best.rationale, best.commands


def _label(unit) -> str:
    from openinterpretability.explain import PoleAttribution

    return PoleAttribution(*unit, 0.0, 0.0).label


def plan(result, states: dict, model: str, *, target: float = TARGET, window=None,
         groups=(), now: datetime | None = None) -> Plan:  # fmt: skip
    """The experiments that raise ``result``'s explained share to ``target`` (module docstring)."""
    now = now or datetime.now(UTC)
    total = sum(share for _, share, _, _ in result.order) or 1.0
    explained = result.verified_share
    slugs = set(states)
    planned_groups = plan_groups(result, groups, states, model, window, slugs, now)
    holders = {u: [g.target for g in planned_groups if g.target in names]
               for u, names in _membership(groups).items()}  # fmt: skip
    experiments, blocked_units, planned, blocked = [], [], 0.0, 0.0
    for unit, raw, status, done in result.order:
        if explained + planned >= target - 1e-12:
            break
        if done or not raw:
            continue
        share, found = raw / total, _unit(model, unit, status, states, window, slugs, now)
        if isinstance(found, str):
            blocked += share
            blocked_units.append({"target": _label(unit), "share": share, "reason": found})
            continue
        planned += share
        entity = "head" if unit[2] == HEAD else "pole"
        held = holders.get(unit, [])
        experiments.append(
            PlannedExperiment(_label(unit), entity, share, status, *found, held, [unit])
        )
    return Plan(target, explained, planned, blocked, experiments, planned_groups, blocked_units)


def _membership(groups) -> dict:
    """Each unit key → the names of the groups that hold it."""
    out: dict = {}
    for g in groups:
        for unit in g.units:
            out.setdefault(unit, set()).add(g.name)
    return out


def plan_groups(result, groups, states, model, window, slugs, now) -> list[PlannedExperiment]:
    """An experiment for each superposition and circuit that carries the answer, unverified."""
    by_name = {(g.kind, g.name): g for g in groups}
    total = result.total or 1.0
    out = []
    for found in result.groups:
        if found.status == "VERIFIED" or not found.attribution:
            continue
        g = by_name[(found.kind, found.name)]
        make = _superposition if g.kind == "superposition" else _circuit
        draw, why, commands = make(g, found.status, states, model, window, slugs, now)
        share = abs(found.attribution) / total
        out.append(PlannedExperiment(g.name, g.kind, share, found.status, draw, why, commands,
                                     units=list(g.units)))  # fmt: skip
    return out


def _superposition(g, status, states, model, window, slugs, now):  # noqa: ARG001
    # The same arguments as _circuit: plan_groups calls either one through ``make``.
    from openinterpretability.planner_superpositions import (
        _candidate,
        default_definitions,
        definition_identifier,
    )

    state = states.get(g.slug)
    best = _follow_up(state, slugs, now) if state is not None else None
    if best is None:
        ident = definition_identifier(g.definition)
        best = _candidate(ident, g.definition, default_definitions(model), window)
    return best.kind, best.rationale, best.commands


def _circuit(g, status, states, model, window, slugs, now):  # noqa: ARG001
    # The same arguments as _superposition: plan_groups calls either one through ``make``.
    from openinterpretability.core.paths import DEFAULT_EXCHANGE_DIR

    files = f"{DEFAULT_EXCHANGE_DIR}/circuit-{g.name}"  # a working folder, not a cited input
    why = (
        f"{status}: a task hypothesis on this prompt. One prompt has no corrupted twin, so "
        f"write {files}/task.json first: this prompt as a clean prompt, with a corrupted "
        "prompt of the same token length"
    )
    commands = [
        f"open-interpretability circuits propose --model {model} --circuit {g.name} "
        f"--task {files}/task.json --explanation '<what the members compute>' "
        f"--agent-model <model> --out {files}/hypothesis.json",
        f"open-interpretability circuits review --hypothesis {files}/hypothesis.json "
        f"--probes {files}/probes.json",
    ]
    return "circuit", why, commands


def answer_step(wiki_dir, model: str, prompt: str, position: int, answer: str,
                coverage: dict, dependence: dict | None = None) -> dict:  # fmt: skip
    """The answer experiment that the plan lists last (#4778, ``core/answer_records.py``).

    Units say what each part of the model does. An answer claim says how they make this
    answer. So it is planned once some units are explained, which gives a minimal set. The
    attempts go on until one claim is VERIFIED, or ``MAX_ATTEMPTS`` claims are on record.

    ``dependence`` is :func:`~openinterpretability.answer_review.dependence`. When it says
    that the all-unit cut is not measurable, no attempt is planned, because no route can
    pass ``necessity`` and ``sufficiency`` (#5967).
    """
    from openinterpretability.core.answer_records import (
        MAX_ATTEMPTS,
        load,
        next_attempt,
        summary,
        working_dir,
    )
    from openinterpretability.core.explanation_records import prompt_id

    pid = prompt_id(model, position, prompt)
    records = load(wiki_dir, model, pid)
    newest = records[-1].verdict.status if records else "unstudied"
    out = {"prompt_id": pid, "status": newest, "attempts": len(records), "commands": [],
           "newest": summary(records[-1]) if records else None,
           "dependence": dependence}  # fmt: skip
    attempt = next_attempt(records)
    if newest == "VERIFIED":
        return out | {"reason": "a claim is VERIFIED"}
    if attempt is None:
        return out | {"reason": f"blocked: {MAX_ATTEMPTS} claims, none VERIFIED"}
    if not coverage.get("minimal_units"):
        return out | {"reason": "no explained units yet: explain the units first"}
    if dependence is not None and not dependence["measurable"]:
        return out | {"reason": _cancel_reason(dependence)}
    folder = working_dir(pid).as_posix()
    out["commands"] = [
        f"open-interpretability answers propose-prompt {shlex.quote(prompt)} --model {model} "
        f"--position {position} --answer {shlex.quote(answer)}",
        f"open-interpretability answers review-prompt --claim {folder}/claim-{attempt}.json",
        f"open-interpretability answers review --claim {folder}/claim-{attempt}.json "
        f"--probes {folder}/answer-{attempt}.probes.json",
    ]
    why = (f"attempt {attempt} of at most {MAX_ATTEMPTS}: the answer-proposer and "
           "answer-reviewer agents write the claim and the probes")  # fmt: skip
    return out | {"reason": why}


def _cancel_reason(dependence: dict) -> str:
    """Why :func:`answer_step` plans no attempt when the units' effects cancel (#5967).

    Args:
        dependence: The output of :func:`~openinterpretability.answer_review.dependence`.

    Returns:
        The reason, with the all-unit cut and the largest absolute cut of a random set.
    """
    largest = max((abs(c) for c in dependence["null_cuts"]), default=0.0)
    return (f"not planned: ablating every unit cuts the answer's log-probability by "
            f"{dependence['all_cut']:.3f}, no more than a random set of the minimal set's "
            f"size ({largest:.3f}). The units' effects cancel, so the gates necessity and "
            "sufficiency have no baseline (#5967)")  # fmt: skip


def answer_lines(step: dict) -> list[str]:
    """How the plan prints :func:`answer_step`."""
    lines = [f"Answer experiment: {step['status']}, {step['attempts']} claims; {step['reason']}."]
    return lines + [f"    $ {command}" for command in step["commands"]]
