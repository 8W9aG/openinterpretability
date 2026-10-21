from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from openinterp.builder.wiki import write_experiment_page
from openinterp.core.schemas import AuditRecord, ReviewMetrics, ReviewPolicy
from openinterp.planner import BASE_WEIGHTS, claim_key, plan_next

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _record(record, *, status, gates=None, policy=True, idx=117, pole="positive"):
    ident = record.submission.identifier.model_copy(update={"neuron_idx": idx, "pole": pole})
    metrics = None
    if gates is not None:
        metrics = ReviewMetrics(
            specificity=1.0,
            sensitivity=1.0,
            balanced_accuracy=1.0,
            counterfactual_reproduction=1.0,
            lookahead_consistent=True,
            independent_review=gates.get("independent_review", True),
            gates=gates,
        )
    verdict = record.verdict.model_copy(
        update={"status": status, "metrics": metrics, "policy": ReviewPolicy() if policy else None}
    )
    return AuditRecord(
        submission=record.submission.model_copy(update={"identifier": ident}), verdict=verdict
    )


def _registry(tmp_path, *records):
    wiki = tmp_path / "wiki"
    for rec in records:
        write_experiment_page(rec, wiki)
    return wiki


def _kinds(plan):
    return {(c.kind, c.slug.rsplit("/", 1)[-1]) for c in plan.candidates}


def test_empty_registry_explores(tmp_path):
    plan = plan_next(tmp_path / "wiki", seed=1, epsilon=0.0, now=NOW)
    assert plan.explored and plan.choice.kind == "explore"
    assert plan.choice.commands[0].startswith("openinterp audit --model Qwen/Qwen2.5-0.5B --random")


def test_candidates_follow_verdicts(tmp_path, record):
    wiki = _registry(
        tmp_path,
        _record(
            record,
            status="NEEDS_REVISION",
            idx=1,
            gates={"sensitivity": False, "independent_review": False},
        ),
        _record(record, status="REJECTED", idx=2, gates={"specificity": False}),
        _record(record, status="VERIFIED", idx=3, gates={"specificity": True}),
        _record(record, status="VERIFIED", idx=4, policy=False),  # pre-0.2 verdict
    )
    plan = plan_next(wiki, seed=1, epsilon=0.0, now=NOW)
    kinds = _kinds(plan)
    assert {("revise", "N1-positive"), ("independent_review", "N1-positive")} <= kinds
    assert ("new_hypothesis", "N2-positive") in kinds  # a rejection is still followed up
    assert ("replicate", "N3-positive") in kinds
    assert ("reevaluate", "N4-positive") in kinds
    assert ("opposite_pole", "N3-negative") in kinds
    revise = next(c for c in plan.candidates if c.kind == "revise")
    assert "sensitivity" in revise.rationale
    supersedes = revise.commands[0].split(" --pole positive --supersedes ")[1]
    assert supersedes.endswith("N1-positive/20261001T120000Z.json")


def test_opposite_pole_skipped_when_studied(tmp_path, record):
    wiki = _registry(
        tmp_path,
        _record(record, status="VERIFIED", idx=5, gates={}),
        _record(record, status="VERIFIED", idx=5, pole="negative", gates={}),
    )
    assert not any(c.kind == "opposite_pole" for c in plan_next(wiki, seed=1, now=NOW).candidates)


def test_decay_never_reaches_zero(tmp_path, record):
    rec = _record(record, status="REJECTED", idx=7, gates={})
    claimed_recent = NOW - timedelta(hours=1)
    rec = rec.model_copy(
        update={"verdict": rec.verdict.model_copy(update={"timestamp": claimed_recent})}
    )
    wiki = _registry(tmp_path, rec)
    claims = [f"experiment: {claim_key(rec.submission.identifier.slug)}"]
    plan = plan_next(wiki, seed=1, epsilon=0.0, claims=claims, now=NOW)
    new_hyp = next(c for c in plan.candidates if c.kind == "new_hypothesis")
    assert 0 < new_hyp.weight < BASE_WEIGHTS["new_hypothesis"] * 0.02  # recency × claim
    assert "open PR" in new_hyp.rationale and "6h" in new_hyp.rationale


def test_seed_reproduces_choice(tmp_path, record):
    wiki = _registry(tmp_path, _record(record, status="NEEDS_REVISION", gates={}))
    a = plan_next(wiki, seed=42, now=NOW)
    b = plan_next(wiki, seed=42, now=NOW)
    assert (a.choice, a.explored) == (b.choice, b.explored)


def test_sampling_spreads_agents_out(tmp_path, record):
    """Independent agents must not all pick the same experiment."""
    wiki = _registry(
        tmp_path,
        _record(record, status="NEEDS_REVISION", idx=1, gates={"independent_review": False}),
        _record(record, status="VERIFIED", idx=2, gates={}),
    )
    picks = Counter(
        (p.choice.kind, p.choice.slug)
        for p in (plan_next(wiki, seed=s, now=NOW) for s in range(300))
    )
    explored = picks.pop(("explore", None)) / 300
    # Each explore draws its own random neuron, so only follow-ups can pile up.
    top_share = picks.most_common(1)[0][1] / picks.total()
    assert len(picks) >= 4 and top_share < 0.5
    assert 0.6 < explored < 0.8  # epsilon = 0.7


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_epsilon_extremes(tmp_path, record, epsilon):
    wiki = _registry(tmp_path, _record(record, status="VERIFIED", gates={}))
    plans = [plan_next(wiki, seed=s, epsilon=epsilon, now=NOW) for s in range(20)]
    assert all(p.explored == (epsilon == 1.0) for p in plans)


def test_cli_next(tmp_path, record):
    import json

    from click.testing import CliRunner

    from openinterp.cli import cli

    wiki = _registry(tmp_path, _record(record, status="NEEDS_REVISION", gates={}))
    args = ["next", "--wiki-dir", str(wiki), "--no-claims"]
    result = CliRunner().invoke(cli, [*args, "--seed", "3", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["seed"] == 3
    text = CliRunner().invoke(cli, [*args, "--top", "3"]).output
    assert "Next experiment:" in text and "seed" in text


def test_latest_experiment_drives_follow_ups(tmp_path, record):
    older = _record(record, status="NEEDS_REVISION", gates={"sensitivity": False}, idx=1)
    newer = _record(record, status="VERIFIED", gates={}, idx=1)
    newer = newer.model_copy(
        update={
            "verdict": newer.verdict.model_copy(
                update={"timestamp": older.verdict.timestamp + timedelta(hours=1)}
            )
        }
    )
    wiki = _registry(tmp_path, newer, older)
    plan = plan_next(wiki, seed=1, epsilon=0.0, now=NOW)
    kinds = {kind for kind, _ in _kinds(plan)}
    assert "revise" not in kinds and "replicate" in kinds
    assert all("2 experiments so far" in c.rationale for c in plan.candidates)
