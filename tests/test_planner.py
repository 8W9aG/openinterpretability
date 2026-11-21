import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from openinterpretability import planner, planner_claims, planner_models
from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.core.licenses import ModelLicense
from openinterpretability.core.schemas import (
    BUILTIN_CORPUS,
    AuditRecord,
    ExperimentProvenance,
    Histogram,
    ReviewMetrics,
    ReviewPolicy,
)
from openinterpretability.planner import (
    BASE_WEIGHTS,
    claim_key,
    configured_licenses,
    configured_models,
    plan_next,
)

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_cached_configs(monkeypatch):
    """Model sizes come from the records only, whatever the Hugging Face cache holds."""
    monkeypatch.setattr(planner_models, "model_dims", lambda model, download=True: None)


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
    assert plan.choice.commands[0].startswith(
        "open-interpretability audit --model Qwen/Qwen2.5-0.5B --random"
    )


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


def _default_on_histogram(rec, active=90):
    """Return ``rec`` with ``active`` of 100 scan tokens above the activity line."""
    line = ReviewPolicy().activation_threshold * rec.submission.max_activation
    histogram = Histogram(bin_edges=[0.0, line, 3.0], counts=[100 - active, active])
    return rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"activation_histogram": histogram})}
    )


@pytest.mark.parametrize("status", ["NEEDS_REVISION", "REJECTED", "VERIFIED"])
def test_a_default_on_histogram_ends_follow_ups_without_the_gate(tmp_path, record, status):
    # A protocol 0.7 record has no not_default_on gate. Only its histogram shows the state.
    gates = {"specificity": status == "VERIFIED"}
    rec = _default_on_histogram(_record(record, status=status, idx=8, gates=gates))
    plan = plan_next(_registry(tmp_path, rec), seed=1, epsilon=0.0, now=NOW)
    assert _kinds(plan) == {("opposite_pole", "N8-negative")}


def test_a_default_on_histogram_keeps_reevaluate(tmp_path, record):
    rec = _record(record, status="VERIFIED", idx=8, gates={}, policy=False)
    plan = plan_next(_registry(tmp_path, _default_on_histogram(rec)), seed=1, epsilon=0.0, now=NOW)
    assert _kinds(plan) == {("reevaluate", "N8-positive"), ("opposite_pole", "N8-negative")}
    weights = {c.kind: c.weight for c in plan.candidates}
    assert weights == {
        "reevaluate": BASE_WEIGHTS["reevaluate"],
        "opposite_pole": BASE_WEIGHTS["opposite_pole"],
    }


def test_default_on_histograms_never_draw_a_scan_that_must_stop(tmp_path, record):
    wiki = _registry(
        tmp_path,
        _default_on_histogram(_record(record, status="NEEDS_REVISION", idx=1, gates={})),
        _default_on_histogram(_record(record, status="REJECTED", idx=2, gates={})),
        _default_on_histogram(_record(record, status="NEEDS_REVISION", idx=3, gates={}), active=49),
    )
    drawn = Counter(
        (c.kind, c.slug.rsplit("/", 1)[-1])
        for c in (plan_next(wiki, seed=s, epsilon=0.0, now=NOW).choice for s in range(200))
    )
    assert ("revise", "N3-positive") in drawn  # below the line: still auditable
    assert not {key for key in drawn if key[1] in {"N1-positive", "N2-positive"}}


def test_a_builtin_sample_histogram_does_not_settle_default_on(tmp_path, record):
    rec = _default_on_histogram(_record(record, status="REJECTED", idx=9, gates={}))
    plan = plan_next(_registry(tmp_path, _scanned_on(rec, BUILTIN_CORPUS)), seed=1, now=NOW)
    assert _kinds(plan) == {("rescan", "N9-positive"), ("opposite_pole", "N9-negative")}


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
    assert rescan.commands[0].startswith("open-interpretability audit --model ")
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

    from openinterpretability.cli import cli

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
    """With no model size known, a model with no records gets most explore draws."""
    wiki = _registry(tmp_path, *(_record(record, status="VERIFIED", idx=i) for i in range(8)))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["Qwen/Qwen2.5-0.5B", "new/model"]\n'
    )
    picks = _explored_models(wiki)
    # Weights 1/√9 and 1/√1, so the new model gets 75% of explore draws.
    assert 0.68 < picks["new/model"] / picks.total() < 0.82
    assert picks["Qwen/Qwen2.5-0.5B"] > 0
    plan = plan_next(wiki, seed=1, epsilon=1.0, now=NOW)
    assert plan.explore_models == pytest.approx({"Qwen/Qwen2.5-0.5B": 0.25, "new/model": 0.75})
    assert plan.focus is None and "by coverage" in plan.choice.rationale


def _claims(model_slug, n):
    return [f"experiment: {model_slug}-L0-N{i}-positive" for i in range(n)]


def _on(record, model, idx, status):
    """A record of ``model``'s L3/N``idx``+ with ``status``."""
    rec = _record(record, status=status, idx=idx)
    ident = rec.submission.identifier.model_copy(update={"model_name": model})
    return rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"identifier": ident})}
    )


def test_smallest_model_below_target_is_the_focus(tmp_path, record, monkeypatch):
    """The smallest model gets most draws until 80% of it is verified; then the next does."""
    sizes = {"small/m": (2, 10), "mid/m": (4, 100), "big/m": (8, 100)}
    monkeypatch.setattr(planner_models, "model_dims", lambda model, download=True: sizes[model])
    for d in (tmp_path, tmp_path / "a", tmp_path / "b"):
        d.mkdir(exist_ok=True)
        (d / "pyproject.toml").write_text(
            '[tool.openinterpretability]\nmodels = ["big/m", "mid/m", "small/m"]\n'
        )
    plan = plan_next(tmp_path / "wiki", seed=1, epsilon=1.0, now=NOW, claims=_claims("mid--m", 3))
    assert plan.focus == "small/m"
    # The others share 20%, by 1/√(1 + covered): 1/√4 for mid/m, 1/√1 for big/m.
    assert plan.explore_models == pytest.approx(
        {"small/m": 0.8, "mid/m": 0.2 / 3, "big/m": 0.4 / 3}
    )
    picks = _explored_models(tmp_path / "wiki", n=200)
    assert 0.72 < picks["small/m"] / picks.total() < 0.88
    seed = next(
        s
        for s in range(50)
        if plan_next(tmp_path / "wiki", seed=s, epsilon=1.0, now=NOW).choice.model == "small/m"
    )
    rationale = plan_next(tmp_path / "wiki", seed=seed, epsilon=1.0, now=NOW).choice.rationale
    assert "small/m drawn with probability 80% by the smallest model below" in rationale

    # Every neuron studied, 15 of 20 verified: still below the target.
    statuses = ["VERIFIED"] * 15 + ["REJECTED"] * 5
    recs = [_on(record, "small/m", i, st) for i, st in enumerate(statuses)]
    plan = plan_next(_registry(tmp_path / "a", *recs), seed=1, epsilon=1.0, now=NOW)
    assert plan.coverage["small/m"].covered == 20 and plan.coverage["small/m"].verified == 15
    assert plan.focus == "small/m"

    recs = [_on(record, "small/m", i, "VERIFIED") for i in range(16)]  # 16 of 20: the target
    plan = plan_next(_registry(tmp_path / "b", *recs), seed=1, epsilon=1.0, now=NOW)
    assert plan.coverage["small/m"].verified_share == pytest.approx(0.8)
    assert plan.focus == "mid/m" and plan.explore_models["mid/m"] == pytest.approx(0.8)


def test_open_claims_count_as_coverage(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["a/x", "b/y"]\n'
    )
    claims = [
        f"experiment: a--x-L{i}-N1-positive experiment/a--x-L{i}-N1-positive" for i in range(3)
    ]
    plan = plan_next(tmp_path / "wiki", seed=1, now=NOW, claims=claims)
    assert plan.coverage["a/x"].covered == 3  # each claim names its branch twice
    assert plan.explore_models == pytest.approx({"a/x": 1 / 3, "b/y": 2 / 3})  # 1/√4 vs 1/√1


def test_coverage_target_and_share_are_configurable(tmp_path, record):
    """Records give the size; ``[tool.openinterpretability]`` can move the target and the share."""
    sub = record.submission.model_copy(update={"model_num_layers": 2, "model_mlp_width": 10})
    rec = record.model_copy(update={"submission": sub})
    wiki = _registry(tmp_path, *(_record(rec, status="VERIFIED", idx=i) for i in range(3)))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["Qwen/Qwen2.5-0.5B", "new/model"]\n'
        "coverage_target = 0.5\nothers_share = 0.1\n"
    )
    plan = plan_next(wiki, seed=1, epsilon=1.0, now=NOW)
    assert plan.coverage["Qwen/Qwen2.5-0.5B"].total == 20
    assert plan.focus == "Qwen/Qwen2.5-0.5B"  # 3 of 20 verified is below 50%
    assert plan.explore_models == pytest.approx({"Qwen/Qwen2.5-0.5B": 0.9, "new/model": 0.1})


def test_model_option_overrides_config(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.openinterpretability]\nmodels = ["a/x"]\n')
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


def test_every_configured_model_has_a_license():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert set(configured_licenses(pyproject)) >= set(configured_models(pyproject))


def test_configured_licenses_parse_and_reject_typos(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    assert configured_licenses(pyproject) == {}
    terms = 'license = "x"\nurl = "https://u"\ngated = true\npublishing_allowed = false\n'
    pyproject.write_text(f'[tool.openinterpretability.licenses."a/x"]\n{terms}')
    assert configured_licenses(pyproject)["a/x"] == ModelLicense(
        license="x", url="https://u", gated=True, publishing_allowed=False
    )
    pyproject.write_text(f'[tool.openinterpretability.licenses."a/x"]\n{terms}publishable = true\n')
    with pytest.raises(ValidationError):
        configured_licenses(pyproject)


def test_short_context_models_are_scanned_in_windows(tmp_path, record):
    """GPT-2 is scanned in windows.

    Its explore draws, and follow-ups on a record scanned in windows, rescan with the
    window length.
    """
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["a/x"]\n'
        '[tool.openinterpretability.window_tokens]\n"a/x" = 128\n'
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
    audits = [
        cmd for c in plan.candidates for cmd in c.commands if "open-interpretability audit" in cmd
    ]
    assert audits and all("--window-tokens 256" in cmd for cmd in audits)


def test_stopped_scans_count_as_covered_and_only_proofs_as_verified():
    from openinterpretability.core.identifiers import NeuronIdentifier
    from openinterpretability.planner_models import model_coverage

    stopped = [
        NeuronIdentifier(model_name="a/x", layer=0, neuron_idx=i, pole="positive") for i in range(3)
    ]
    coverage = model_coverage(["a/x"], [], [], dead=stopped[:1], stopped=stopped)["a/x"]
    assert coverage.covered == 3 and coverage.verified == 1


def test_fork_prs_claim_only_by_an_experiment_branch(monkeypatch):
    prs = [
        {
            "title": "Mine: L1/N2",
            "headRefName": "experiment/m-L1-N2-positive",
            "isCrossRepository": False,
        },
        {"title": "x L3/N4 L5/N6 L7/N8", "headRefName": "patch-1", "isCrossRepository": True},
        {
            "title": "x L9/N9",
            "headRefName": "experiment/m-L9-N9-positive",
            "isCrossRepository": True,
        },
    ]
    monkeypatch.setattr(planner_claims.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        planner_claims.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(prs)),
    )
    assert planner.open_pr_claims() == [
        "Mine: L1/N2 experiment/m-L1-N2-positive",
        "m-L9-N9-positive",
    ]


def _fork_prs(monkeypatch, *branches):
    prs = [{"title": "x", "headRefName": b, "isCrossRepository": True} for b in branches]
    monkeypatch.setattr(planner_claims.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        planner_claims.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(prs)),
    )
    return planner.open_pr_claims()


def test_a_fork_branch_that_joins_several_keys_claims_none(monkeypatch):
    joined = "experiment/" + "-".join(f"a--x-L{i}-N{i}-positive" for i in (1, 2, 3))
    claims = _fork_prs(monkeypatch, joined)
    assert claims == []
    coverage = planner_models.model_coverage(["a/x"], [], claims, dead=[], stopped=[])
    assert coverage["a/x"].covered == 0


def test_a_fork_claim_matches_only_an_equal_key(monkeypatch):
    claims = _fork_prs(
        monkeypatch, "experiment/Qwen--Qwen2.5-0.5B-L3-H12", "experiment/m-L1-N2-positive-rescan"
    )
    assert claims == ["Qwen--Qwen2.5-0.5B-L3-H12", "m-L1-N2-positive"]
    assert planner.claimed("Qwen--Qwen2.5-0.5B-L3-H12", claims)
    assert not planner.claimed("Qwen--Qwen2.5-0.5B-L3-H1", claims)
    assert planner.claimed("m-L1-N2-positive", claims)
    # A PR from this repository still claims each key that its text contains.
    assert planner.claimed("m-L3-H1", ["Study m-L3-H12 and m-L3-H1"])


def _probes_of_its_own(rec, tag, status):
    """``rec`` with a status and a probe set of its own, so it is not a re-measurement."""
    tests = [t.model_copy(update={"prompt": f"{t.prompt} {tag}"})
             for t in rec.verdict.adversarial_test_results]  # fmt: skip
    verdict = rec.verdict.model_copy(update={"status": status, "adversarial_test_results": tests})
    return rec.model_copy(update={"verdict": verdict})


def test_an_even_split_draws_a_resolving_review(tmp_path, record):
    """One VERIFIED and one REJECTED independent reading, scored alike: CONTESTED, so `resolve`.

    The issues are #3212 and #4900. A NEEDS_REVISION reading would not contest the VERIFIED one.
    """
    rec = _record(record, status="VERIFIED", idx=1, gates={})
    split = [
        _probes_of_its_own(_reviewed_by(rec, "x/a"), "a", "VERIFIED"),
        _probes_of_its_own(_reviewed_by(rec, "x/b", hours=1), "b", "REJECTED"),
    ]
    plan = plan_next(_registry(tmp_path, *split), seed=1, now=NOW)
    assert ("resolve", "N1-positive") in _kinds(plan)
    assert ("revise", "N1-positive") not in _kinds(plan)


def test_a_failed_rival_of_a_verified_hypothesis_draws_a_revision(tmp_path, record):
    """VERIFIED, then a new hypothesis REJECTED: CONTESTED, so `revise`, not `resolve` (#4654)."""
    rec = _record(record, status="VERIFIED", idx=1, gates={})
    sub = rec.submission.model_copy(update={"natural_language_explanation": "A rival."})
    verdict = rec.verdict.model_copy(
        update={"status": "REJECTED", "timestamp": rec.verdict.timestamp + timedelta(hours=2)}
    )
    rival = _probes_of_its_own(rec.model_copy(update={"submission": sub, "verdict": verdict}),
                               "r", "REJECTED")  # fmt: skip
    plan = plan_next(_registry(tmp_path, rec, rival), seed=1, now=NOW)
    revise = [c for c in plan.candidates
              if (c.kind, c.slug.rsplit("/", 1)[-1]) == ("revise", "N1-positive")]  # fmt: skip
    assert revise and "submit one that addresses both" in revise[0].rationale
    assert ("resolve", "N1-positive") not in _kinds(plan)


def test_a_unit_studied_in_the_last_day_is_drawn_at_half_weight(tmp_path, record):
    rec = _record(record, status="REJECTED", idx=7, gates={})
    twelve_hours_ago = NOW - timedelta(hours=12)
    rec = rec.model_copy(
        update={"verdict": rec.verdict.model_copy(update={"timestamp": twelve_hours_ago})}
    )
    plan = plan_next(_registry(tmp_path, rec), seed=1, epsilon=0.0, now=NOW)
    new_hyp = next(c for c in plan.candidates if c.kind == "new_hypothesis")
    assert new_hyp.weight == pytest.approx(BASE_WEIGHTS["new_hypothesis"] * 0.5)
    assert "studied in the last 24h" in new_hyp.rationale and "6h" not in new_hyp.rationale


def test_an_explore_draw_can_be_weighted_by_effect(tmp_path):
    from openinterpretability.planner_effects import table_path

    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["toy/model"]\nhead_share = 0.0\n'
        "effect_share = 1.0\n"
    )
    table = table_path(tmp_path, "toy/model")
    table.parent.mkdir(parents=True)
    rows = [{"layer": 2, "neuron": 9, "pole": "negative", "effect": 0.5}]
    table.write_text(json.dumps({"poles": rows}))
    plan = plan_next(tmp_path / "wiki", seed=1, epsilon=1.0, now=NOW)
    assert plan.explored and plan.choice.slug.endswith("L2/N9-negative")
    assert "--selection attribution" in plan.choice.commands[0]
