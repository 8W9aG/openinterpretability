"""Circuit experiments: a task hypothesis about a circuit, its review and its verdict (#4284).

A circuit is a cluster of a model's wiring graph with an id that survives as the registry
grows (:mod:`openinterpretability.circuit_ids`). An experiment on it tests one **task hypothesis**:
a behaviour, the members that carry it, and each member's role. The proposer gives the
task's prompt pairs. The reviewer gives holdout pairs of the same task, and other tasks
that the circuit should not carry. The review measures each set with mean ablation of
neurons and heads (:mod:`openinterpretability.harness.circuit_measures`), and :func:`decide`
judges the measures against :class:`CircuitPolicy`.

Circuit verdicts have their own rule version, :data:`CIRCUIT_PROTOCOL_VERSION`, so the
rule for neuron poles, heads and latents (``PROTOCOL_VERSION``) is unchanged
(``docs/PROTOCOL.md`` §6). Records live in ``wiki/circuits/<model slug>/<circuit id>/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from openinterpretability.core.strict import StrictModel

CIRCUIT_PROTOCOL_VERSION = "1.0"
CIRCUITS_WIKI = "circuits"  # records live in wiki/circuits/
CircuitStatus = Literal["VERIFIED", "REJECTED", "NEEDS_REVISION"]


class PromptPair(StrictModel):
    """A clean prompt and a corrupted one of equal token length, with the token each predicts."""

    clean: str = Field(min_length=1)
    corrupted: str = Field(min_length=1)
    good: str = Field(min_length=1, description="The clean prompt's next token, as text.")
    bad: str = Field(min_length=1, description="The corrupted prompt's next token, as text.")


class CircuitTask(StrictModel):
    """A behaviour, measured as logit(good) − logit(bad) at each pair's last position."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    pairs: list[PromptPair] = Field(min_length=1)


class CircuitHypothesis(StrictModel):
    """What a circuit is claimed to do on a task, and which members carry it."""

    model_name: str
    model_revision: str | None = None
    circuit_id: str = Field(pattern=r"^C\d+$")
    task: CircuitTask
    members: list[str] = Field(min_length=1, description="Neurons and heads: L4/N2999, L7/H2.")
    cluster_members: list[str] = Field(
        default_factory=list,
        description="The circuit's members in data/circuits when the hypothesis was made, "
        "so a page can tell when the circuit has moved on (stale).",
    )
    roles: dict[str, str] = Field(default_factory=dict, description="Member → its role.")
    explanation: str = Field(min_length=1, description="What the members compute together.")
    submitting_agent_model: str
    supersedes: str | None = None


class CircuitProbes(StrictModel):
    """The reviewer's tests: holdout pairs of the task, and tasks it should not carry."""

    holdouts: list[PromptPair] = Field(min_length=1)
    negative_tasks: list[CircuitTask] = Field(min_length=1)
    reviewer_agent_model: str


class CircuitPolicy(StrictModel):
    """The gates of a circuit verdict (``docs/PROTOCOL.md`` §4, from #4283)."""

    min_necessity: float = 0.02  # the effect must be large enough to matter
    null_quantile: float = 0.95  # and beat this quantile of random sets like the circuit
    null_sets: int = 64
    max_negative_necessity: float = 0.2  # a negative task passes below this loss
    min_specificity: float = 0.8  # the share of negative tasks that must pass
    protocol_version: str = CIRCUIT_PROTOCOL_VERSION


class CircuitMeasures(StrictModel):
    """What the review measured. Each value is relative to ablating every neuron and head."""

    holdout_faithfulness: float
    holdout_necessity: float
    null_necessities: list[float] = Field(description="Random sets like the members, sorted.")
    negative_necessities: dict[str, float] = Field(description="Task name → its necessity.")
    member_necessities: dict[str, float] = Field(default_factory=dict)


class CircuitVerdict(StrictModel):
    """The decision, with the gates it rests on."""

    status: CircuitStatus
    gates: dict[str, bool]
    policy: CircuitPolicy
    measures: CircuitMeasures
    reviewer_agent_model: str
    timestamp: datetime


class CircuitRecord(StrictModel):
    """One circuit experiment, archived at ``wiki/circuits/<model>/<id>/<experiment>.json``."""

    hypothesis: CircuitHypothesis
    probes: CircuitProbes
    verdict: CircuitVerdict


def null_threshold(nulls: list[float], quantile: float) -> float:
    """The ``quantile`` of the sorted null necessities, or 0 for none."""
    if not nulls:
        return 0.0
    ordered = sorted(nulls)
    return ordered[max(0, min(len(ordered) - 1, int(quantile * len(ordered)) - 1))]


def decide(
    measures: CircuitMeasures, policy: CircuitPolicy
) -> tuple[CircuitStatus, dict[str, bool]]:
    """The status and the gates from the measures.

    VERIFIED when every gate passes. NEEDS_REVISION when the members carry the task but
    also carry other tasks: the members or the claim are too broad. REJECTED otherwise.
    """
    negatives = list(measures.negative_necessities.values())
    passing = sum(abs(n) < policy.max_negative_necessity for n in negatives)
    specificity = passing / len(negatives) if negatives else 0.0
    gates = {
        "necessity_floor": measures.holdout_necessity >= policy.min_necessity,
        "beats_null": measures.holdout_necessity
        > null_threshold(measures.null_necessities, policy.null_quantile),
        "specificity": specificity >= policy.min_specificity,
    }
    if all(gates.values()):
        return "VERIFIED", gates
    if gates["necessity_floor"] and gates["beats_null"]:
        return "NEEDS_REVISION", gates
    return "REJECTED", gates


def record_path(wiki_dir: Path, record: CircuitRecord) -> Path:
    """``wiki/circuits/<model slug>/<circuit id>/<UTC timestamp>.json``."""
    h, when = record.hypothesis, record.verdict.timestamp.astimezone(UTC)
    slug = h.model_name.replace("/", "--")
    return wiki_dir / CIRCUITS_WIKI / slug / h.circuit_id / f"{when:%Y%m%dT%H%M%SZ}.json"
