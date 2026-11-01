import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openinterp.builder.wiki import write_experiment_page
from openinterp.core.schemas import (
    BUILTIN_CORPUS,
    AuditRecord,
    ExperimentProvenance,
    ReviewMetrics,
    ReviewPolicy,
)
from openinterp.planner import BASE_WEIGHTS, claim_key, configured_models, plan_next

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


def test_default_on_units_get_no_follow_ups_but_their_other_pole(tmp_path, record):
    gates = {"not_default_on": False, "specificity": False, "independent_review": False}
    wiki = _registry(tmp_path, _record(record, status="NEEDS_REVISION", idx=6, gates=gates))
    kinds = _kinds(plan_next(wiki, seed=1, epsilon=0.0, now=NOW))
    assert kinds == {("opposite_pole", "N6-negative")}


def _scanned_on(rec, corpus):
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus=corpus,
        n_snippets=1,
        tokens_scanned=552,
        selection="random",
        harness_version="t",
    )
    return rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"provenance": provenance})}
    )


@pytest.mark.parametrize(
    ("status", "gates", "policy"),
    [
        ("VERIFIED", {"independent_review": False}, True),  # would be independent_review
        ("VERIFIED", {}, False),  # would be reevaluate
        ("REJECTED", {"specificity": False}, True),  # would be new_hypothesis
        ("NEEDS_REVISION", {"not_default_on": False}, True),  # default-on on 552 tokens
    ],
)
def test_builtin_sample_records_are_rescanned_not_followed_up(
    tmp_path, record, status, gates, policy
):
    rec = _scanned_on(
        _record(record, status=status, idx=7, gates=gates, policy=policy), BUILTIN_CORPUS
    )
    assert rec.scanned_on_builtin_sample
    plan = plan_next(_registry(tmp_path, rec), seed=1, epsilon=0.0, now=NOW)
    assert _kinds(plan) == {("rescan", "N7-positive"), ("opposite_pole", "N7-negative")}
    rescan = next(c for c in plan.candidates if c.kind == "rescan")
    assert rescan.weight == BASE_WEIGHTS["rescan"]
    assert rescan.commands[0].startswith("openinterp audit --model ")
    assert rescan.commands[0].endswith("N7-positive/20261001T120000Z.json")  # --supersedes it
    assert "--supersedes" not in rescan.audit


def test_records_on_a_corpus_are_not_rescanned(tmp_path, record):
    rec = _scanned_on(_record(record, status="VERIFIED", idx=8, gates={}), "corpus.jsonl")
    assert not rec.scanned_on_builtin_sample
    plan = plan_next(_registry(tmp_path, rec), seed=1, epsilon=0.0, now=NOW)
    assert ("replicate", "N8-positive") in _kinds(plan)
    assert not any(c.kind == "rescan" for c in plan.candidates)


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


def _reviewed_by(rec, reviewer, *, hours=0):
    """The same record reviewed by ``reviewer``; self-reviewed if it is the proposer."""
    independent = reviewer != rec.submission.submitting_agent_model
    metrics = rec.verdict.metrics.model_copy(update={"independent_review": independent})
    verdict = rec.verdict.model_copy(
        update={
            "reviewer_agent_model": reviewer,
            "metrics": metrics,
            "timestamp": rec.verdict.timestamp + timedelta(hours=hours),
        }
    )
    return rec.model_copy(update={"verdict": verdict})


def test_self_review_draws_a_re_review_although_the_gate_passes(tmp_path, record):
    # Under the default policy a self-review passes the independent_review gate.
    rec = _record(record, status="NEEDS_REVISION", idx=1, gates={"independent_review": True})
    proposer = rec.submission.submitting_agent_model
    wiki = _registry(tmp_path, _reviewed_by(rec, proposer))
    assert ("independent_review", "N1-positive") in _kinds(plan_next(wiki, seed=1, now=NOW))


def test_replicate_until_another_model_reviews(tmp_path, record):
    """Two reviewer models that each reviewed their own hypothesis are not a replication."""
    rec = _record(record, status="VERIFIED", idx=1, gates={})
    proposer = rec.submission.submitting_agent_model
    other = rec.model_copy(
        update={
            "submission": rec.submission.model_copy(update={"submitting_agent_model": "other/m"})
        }
    )
    selves = [_reviewed_by(rec, proposer), _reviewed_by(other, "other/m", hours=1)]
    plan = plan_next(_registry(tmp_path / "a", *selves), seed=1, now=NOW)
    assert ("replicate", "N1-positive") in _kinds(plan)
    assert ("challenge", "N1-positive") not in _kinds(plan)
    replicate = next(c for c in plan.candidates if c.kind == "replicate")
    assert "self-review" in replicate.rationale

    crossed = [_reviewed_by(rec, proposer), _reviewed_by(rec, "other/m", hours=1)]
    kinds = _kinds(plan_next(_registry(tmp_path / "b", *crossed), seed=1, now=NOW))
    assert ("challenge", "N1-positive") in kinds and ("replicate", "N1-positive") not in kinds


def _explored_models(wiki, n=400, **kwargs):
    plans = (plan_next(wiki, seed=s, epsilon=1.0, now=NOW, **kwargs) for s in range(n))
    return Counter(p.choice.model for p in plans)


def test_explores_configured_models_by_coverage(tmp_path, record):
    """A configured model with no records is explored; well-covered ones still are."""
    wiki = _registry(tmp_path, *(_record(record, status="VERIFIED", idx=i) for i in range(8)))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterp]\nmodels = ["Qwen/Qwen2.5-0.5B", "new/model"]\n'
    )
    picks = _explored_models(wiki)
    # Weights 1/√9 and 1/√1, so the new model gets 75% of explore draws.
    assert 0.68 < picks["new/model"] / picks.total() < 0.82
    assert picks["Qwen/Qwen2.5-0.5B"] > 0
    plan = plan_next(wiki, seed=1, epsilon=1.0, now=NOW)
    assert plan.explore_models == pytest.approx({"Qwen/Qwen2.5-0.5B": 0.25, "new/model": 0.75})
    assert "by coverage" in plan.choice.rationale


def test_open_claims_count_as_coverage(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.openinterp]\nmodels = ["a/x", "b/y"]\n')
    claims = [
        f"experiment: a--x-L{i}-N1-positive experiment/a--x-L{i}-N1-positive" for i in range(3)
    ]
    plan = plan_next(tmp_path / "wiki", seed=1, now=NOW, claims=claims)
    assert plan.explore_models == pytest.approx({"a/x": 1 / 3, "b/y": 2 / 3})  # 1/√4 vs 1/√1


def test_model_option_overrides_config(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.openinterp]\nmodels = ["a/x"]\n')
    assert set(_explored_models(tmp_path / "wiki", n=20, models=["b/y"])) == {"b/y"}
    assert set(_explored_models(tmp_path / "wiki", n=20)) == {"a/x"}


def test_configured_models_match_ci_allow_list():
    """CI must be able to reproduce every record the planner sends agents to make."""
    root = Path(__file__).resolve().parents[1]
    ci = (root / ".github/workflows/ci.yml").read_text()
    allowed = re.search(r"^\s*ALLOWED_MODELS:\s*(.+?)\s*$", ci, re.M).group(1).split()
    configured = configured_models(root / "pyproject.toml")
    assert configured and sorted(configured) == sorted(allowed)
    assert configured_models(root / "missing.toml") == ()


def test_short_context_models_are_scanned_in_windows(tmp_path, record):
    """GPT-2 is scanned in windows: its explore draws, and follow-ups on a record scanned
    in windows, rescan with the window length."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterp]\nmodels = ["a/x"]\n[tool.openinterp.window_tokens]\n"a/x" = 128\n'
    )
    plan = plan_next(tmp_path / "wiki", seed=1, epsilon=1.0, now=NOW)
    assert plan.choice.commands[0].endswith("--window-tokens 128")

    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="c",
        n_snippets=1,
        tokens_scanned=10,
        snippet_tokens=256,
        selection="random",
        harness_version="t",
    )
    rec = _record(record, status="NEEDS_REVISION", gates={"specificity": False})
    rec = rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"provenance": provenance})}
    )
    plan = plan_next(_registry(tmp_path / "b", rec), seed=1, epsilon=0.0, now=NOW)
    audits = [cmd for c in plan.candidates for cmd in c.commands if "openinterp audit" in cmd]
    assert audits and all("--window-tokens 256" in cmd for cmd in audits)
