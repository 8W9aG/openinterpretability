"""Propose and review a circuit's task hypothesis (docs/circuits-and-hubs.md, issue #4284).

* :func:`propose` grows a circuit's wiring cluster by attribution on the task's pairs
  (:func:`~openinterp.harness.circuit_measures.grow`). The hypothesis then names the
  members that carry the task.
* :func:`review` measures the hypothesis on the reviewer's holdouts and other tasks. It
  compares it with random member sets, and judges it with
  :func:`~openinterp.core.circuit_records.decide`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from openinterp.core.circuit_records import (
    CircuitHypothesis,
    CircuitMeasures,
    CircuitPolicy,
    CircuitProbes,
    CircuitRecord,
    CircuitTask,
    CircuitVerdict,
    PromptPair,
    decide,
)
from openinterp.harness import circuit_measures as cm

MAX_MEMBERS_MEASURED = 50  # beyond this, each member's own necessity is not measured


def pair_task(runner, pairs: list[PromptPair]) -> cm.PairTask:
    """The pairs as token ids. Each target is its text's first token."""
    tok = runner.tokenizer
    clean = [tok.encode(p.clean) for p in pairs]
    corrupted = [tok.encode(p.corrupted) for p in pairs]
    lengths = {len(ids) for ids in clean + corrupted}
    if len(lengths) != 1:
        raise ValueError(f"every prompt must have the same token count; got {sorted(lengths)}")
    good = [tok.encode(p.good)[0] for p in pairs]
    bad = [tok.encode(p.bad)[0] for p in pairs]
    return cm.PairTask(clean, corrupted, good, bad)


def propose(
    runner, base: dict, task: CircuitTask, cluster: list[str], size: int
) -> CircuitHypothesis:
    """A hypothesis whose members are ``cluster`` grown by attribution on ``task`` to ``size``.

    ``base`` holds the other fields: model_name, circuit_id, explanation, roles,
    submitting_agent_model and, optionally, model_revision and supersedes.
    """
    pairs = pair_task(runner, task.pairs)
    seed = [cm.Member.parse(m) for m in cluster if cm.MEMBER.search(m)]
    members = cm.grow(seed, cm.attribution(runner, pairs), size)
    return CircuitHypothesis(task=task, members=[str(m) for m in members],
                             cluster_members=sorted(cluster), **base)  # fmt: skip


def review(
    runner,
    hypothesis: CircuitHypothesis,
    probes: CircuitProbes,
    policy: CircuitPolicy,
    seed: int = 0,
) -> CircuitRecord:
    """Measure ``hypothesis`` on ``probes`` and decide it."""
    members = [cm.Member.parse(m) for m in hypothesis.members]
    holdouts = pair_task(runner, probes.holdouts)
    means = cm.task_means(runner, holdouts)
    found = cm.measures(runner, holdouts, means, members)
    nulls = cm.null_necessity(runner, holdouts, means, members, policy.null_sets, seed)
    negatives = {}
    for task in probes.negative_tasks:
        other = pair_task(runner, task.pairs)
        negatives[task.name] = cm.measures(runner, other, cm.task_means(runner, other), members)[
            "necessity"
        ]
    each = {}
    if len(members) <= MAX_MEMBERS_MEASURED:
        each = {str(m): cm.measures(runner, holdouts, means, [m])["necessity"] for m in members}
    measured = CircuitMeasures(
        holdout_faithfulness=found["faithfulness"], holdout_necessity=found["necessity"],
        null_necessities=nulls, negative_necessities=negatives, member_necessities=each,
    )  # fmt: skip
    status, gates = decide(measured, policy)
    verdict = CircuitVerdict(status=status, gates=gates, policy=policy, measures=measured,
                             reviewer_agent_model=probes.reviewer_agent_model,
                             timestamp=datetime.now(UTC))  # fmt: skip
    return CircuitRecord(hypothesis=hypothesis, probes=probes, verdict=verdict)
