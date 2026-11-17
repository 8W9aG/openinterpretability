"""Registry circuits as unit pages, and the facts on hub pages (#4287).

This is step 5 of ``docs/circuits-and-hubs.md``.

A registry circuit has an id that survives as the registry grows (``C3``,
:mod:`openinterpretability.circuit_ids`). Its page shows its members, where it sits, its
lineage, its tasks with their standing and experiments, and its sub-circuits.

An experiment is **stale** when the circuit's members now differ from the members it
recorded (``cluster_members``). Its row names the units that joined or left since.

A task's standing follows §3 step 9 of the protocol, as ``docs/circuits-and-hubs.md`` gives
it. A record that a later one names in ``supersedes`` doesn't count. Independent reviews,
where the reviewer is not the proposer, outweigh self-reviews. The majority decides, and an
even split is :data:`CONTESTED`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.circuit_ids import CIRCUITS_DIR, jaccard
from openinterpretability.core.circuit_records import CIRCUITS_WIKI, CircuitRecord, null_threshold

CONTESTED = "CONTESTED"  # an even split between statuses
_MEMBER = re.compile(r"L(\d+)/([NH])(\d+)")


@dataclass
class CircuitData:
    """What the build reads for one model's registry circuits and hubs."""

    lineage: list[dict] = field(default_factory=list)  # data/circuits/<model>.jsonl
    records: list[tuple[str, CircuitRecord]] = field(default_factory=list)  # (source, record)
    norms: dict = field(default_factory=dict)  # hubs.load_norms


def load_circuit_data(wiki_dir: Path, circuits_dir: Path = CIRCUITS_DIR, hubs_dir=None) -> dict:
    """Each model's :class:`CircuitData`, keyed by model slug. Missing files give nothing."""
    import json

    from openinterpretability.hubs import HUBS_DIR, load_norms

    out: dict[str, CircuitData] = {}
    for path in sorted(Path(circuits_dir).glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        out.setdefault(path.stem, CircuitData()).lineage = [json.loads(x) for x in lines if x]
    for path in sorted((Path(wiki_dir) / CIRCUITS_WIKI).glob("*/*/*.json")):
        record = CircuitRecord.model_validate_json(path.read_text(encoding="utf-8"))
        source = path.relative_to(Path(wiki_dir).parent).as_posix()
        out.setdefault(path.parts[-3], CircuitData()).records.append((source, record))
    for path in sorted(Path(hubs_dir or HUBS_DIR).glob("*.jsonl")):
        out.setdefault(path.stem, CircuitData()).norms = load_norms(path)
    return out


@dataclass
class ExperimentView:
    """One circuit experiment, and how it stands against the circuit's members now."""

    record: CircuitRecord
    source: str  # the record's path in the repository
    joined: list[str] = field(default_factory=list)  # members since it ran
    left: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """The record's name: its UTC timestamp."""
        return Path(self.source).stem

    @property
    def stale(self) -> bool:
        """True when the circuit's members changed since the experiment ran."""
        return bool(self.joined or self.left)

    @property
    def independent(self) -> bool:
        """True when the reviewer is not the proposer."""
        r = self.record
        return r.verdict.reviewer_agent_model != r.hypothesis.submitting_agent_model

    @property
    def null(self) -> float:
        """The necessity that random sets like the members reach, at the policy's quantile."""
        v = self.record.verdict
        return null_threshold(v.measures.null_necessities, v.policy.null_quantile)

    @property
    def specificity(self) -> float | None:
        """The share of negative tasks that the members don't carry."""
        v = self.record.verdict
        negatives = list(v.measures.negative_necessities.values())
        passing = sum(abs(n) < v.policy.max_negative_necessity for n in negatives)
        return passing / len(negatives) if negatives else None


def task_standing(experiments: list[ExperimentView]) -> str | None:
    """A task's standing from its experiments (module docstring), or None with none."""
    replaced = {e.record.hypothesis.supersedes for e in experiments} - {None}
    live = [e for e in experiments if e.id not in replaced and e.source not in replaced]
    counted = [e for e in live if e.independent] or live
    tally = Counter(e.record.verdict.status for e in counted).most_common()
    if not tally:
        return None
    if len(tally) > 1 and tally[0][1] == tally[1][1]:
        return CONTESTED
    return tally[0][0]


@dataclass
class TaskView:
    """One task hypothesis of a circuit, with its standing and experiments, newest first."""

    name: str
    description: str
    standing: str | None
    experiments: list[ExperimentView]


def label(href: str) -> str:
    """A member's label from its page or its name: ``L3/N7`` or ``L5/H1``."""
    found = _MEMBER.search(href)
    return f"L{found[1]}/{found[2]}{found[3]}" if found else href


def tasks_of(records, members: frozenset[str]) -> list[TaskView]:
    """The circuit's tasks, each with its experiments against the ``members`` now."""
    by_task: dict[str, list[ExperimentView]] = {}
    for source, record in sorted(records, key=lambda sr: sr[1].verdict.timestamp, reverse=True):
        then = frozenset(record.hypothesis.cluster_members)
        joined = sorted(label(m) for m in members - then) if then else []
        left = sorted(label(m) for m in then - members)
        view = ExperimentView(record, source, joined, left)
        by_task.setdefault(record.hypothesis.task.name, []).append(view)
    out = []
    for name, views in by_task.items():
        description = views[0].record.hypothesis.task.description
        out.append(TaskView(name, description, task_standing(views), views))
    return out


@dataclass
class LineageStep:
    """One event of a circuit's lineage, with the units that joined or left at it."""

    date: str
    event: str
    joined: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    others: list[str] = field(default_factory=list)  # a split's parts, a merge's sources


def lineage_of(cid: str, events: list[dict]) -> tuple[list[LineageStep], float | None]:
    """The circuit's lineage, and the overlap of its first members with its current ones."""
    steps, first, before = [], None, frozenset()
    for e in events:
        if e["id"] != cid:
            continue
        members = frozenset(e.get("members", before))
        first = members if first is None and "members" in e else first
        joined = sorted(label(m) for m in members - before) if e["event"] == "grow" else []
        left = sorted(label(m) for m in before - members) if e["event"] == "grow" else []
        others = list(e.get("into", e.get("from", [])))
        steps.append(LineageStep(e["date"], e["event"], joined, left, others))
        before = members
    stability = jaccard(first, before) if first is not None else None
    return steps, stability


def family(cid: str, events: list[dict]) -> tuple[list[str], list[str]]:
    """The circuit's sub-circuits (merged into it), and the larger circuits it merged into."""
    subs = sorted({s for e in events if e["event"] == "merge" and e["id"] == cid
                   for s in e.get("from", []) if s != cid})  # fmt: skip
    parents = sorted({e["id"] for e in events if e["event"] == "merge" and e["id"] != cid
                      and cid in e.get("from", [])})  # fmt: skip
    return subs, parents


@dataclass
class MemberView:
    """One member of a circuit: its label, and its page and status when it has a page."""

    label: str
    href: str | None
    status: str | None


@dataclass
class CircuitView:
    """One registry circuit's page."""

    id: str
    href: str
    members: list[MemberView]
    depth: float | None  # mean relative depth of the members, 0 (first layer) to 1 (last)
    position: float | None  # mean neuron index over the MLP width, of the neuron members
    lineage: list[LineageStep]
    stability: float | None  # Jaccard overlap of the first members and the current ones
    tasks: list[TaskView]
    subs: list[str]  # circuits merged into this one
    parents: list[str]  # larger circuits this one merged into
    wiring: object | None  # builder.circuits.Circuit of the edges between members

    @property
    def layers(self) -> tuple[int, int] | None:
        """The first and last layer of the members."""
        found = [int(m[1]) for m in (_MEMBER.search(x.label) for x in self.members) if m]
        return (min(found), max(found)) if found else None

    @property
    def stale(self) -> int:
        """How many of its experiments ran on members that have changed since."""
        return sum(e.stale for t in self.tasks for e in t.experiments)


def _member(href: str, units: dict) -> MemberView:
    from openinterpretability.builder.charts import best_status

    unit = units.get(href)
    status = best_status(unit) if unit is not None and unit.standings else None
    return MemberView(label(href), href if unit is not None else None, status)


def _location(model, members: list[str]) -> tuple[float | None, float | None]:
    """The mean relative depth of ``members``, and the mean position of the neurons."""
    found = [m for m in (_MEMBER.search(x) for x in members) if m]
    if not found:
        return None, None
    depth = sum(int(m[1]) for m in found) / len(found) / max(1, model.n_layers - 1)
    neurons = [int(m[3]) for m in found if m[2] == "N"]
    width = getattr(model, "mlp_width", None)
    position = sum(neurons) / len(neurons) / width if neurons and width else None
    return depth, position


def circuit_views(model, data: CircuitData, edges: list) -> list[CircuitView]:
    """The model's registry circuits, by id, with ``edges`` (``circuit_edges``) among members."""
    from openinterpretability.builder.circuits import _layout
    from openinterpretability.builder.hrefs import circuit_href
    from openinterpretability.circuit_ids import load_events

    circuits = _kept(data.lineage, load_events(data.lineage))
    units = {u.href: u for layer in model.layers for u in layer.units}
    out = []
    for cid, members in sorted(circuits.items(), key=lambda kv: int(kv[0][1:])):
        inside = [e for e in edges if e.source.href in members and e.target.href in members]
        records = [(s, r) for s, r in data.records if r.hypothesis.circuit_id == cid]
        steps, stability = lineage_of(cid, data.lineage)
        subs, parents = family(cid, data.lineage)
        ordered = sorted(members, key=lambda h: (_sort_key(label(h)), h))
        out.append(CircuitView(
            cid, circuit_href(model.slug, cid), [_member(h, units) for h in ordered],
            *_location(model, ordered), steps, stability, tasks_of(records, members),
            subs, parents, _layout(inside) if inside else None,
        ))  # fmt: skip
    return out


def _sort_key(text: str) -> tuple:
    found = _MEMBER.search(text)
    return (int(found[1]), found[2] == "N", int(found[3])) if found else (0, True, 0)


def _kept(events: list[dict], current: dict[str, frozenset]) -> dict[str, frozenset]:
    """The current circuits, and those merged into another, which keep their pages.

    A circuit merged into another keeps its last members. A sub-circuit keeps its own
    id, tasks and experiments (``docs/circuits-and-hubs.md``, *Sub-circuits*).
    """
    last = {e["id"]: frozenset(e["members"]) for e in events if "members" in e}
    merged = {s for e in events if e["event"] == "merge" for s in e.get("from", [])}
    return current | {cid: last[cid] for cid in merged - set(current) if cid in last}
