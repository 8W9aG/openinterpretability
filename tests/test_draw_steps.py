from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from openinterp import draw_steps
from openinterp.builder.wiki import write_experiment_page
from openinterp.core.schemas import (
    BUILTIN_CORPUS,
    ExperimentProvenance,
    ReviewMetrics,
    ReviewPolicy,
)
from openinterp.draw_steps import random_neuron, render, steps
from openinterp.evidence import ScanRequest, pick_unit
from openinterp.planner import plan_next

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("seed", [0, 1, 953380901, 3280387012])
@pytest.mark.parametrize("pole", [None, "positive"])
def test_random_neuron_matches_the_audits_draw(seed, pole):
    runner = SimpleNamespace(n_layers=24, mlp_width=lambda layer: 4864)
    request = ScanRequest(model_name="Qwen/Qwen2.5-0.5B", random_pick=True, seed=seed, pole=pole)
    ident, _ = pick_unit(runner, request, None)
    assert random_neuron("Qwen/Qwen2.5-0.5B", seed, (24, 4864), pole) == ident.slug


def _registry(tmp_path, record, status):
    gates = {"sensitivity": False} if status == "NEEDS_REVISION" else {}
    metrics = ReviewMetrics(
        specificity=1.0,
        sensitivity=0.5,
        balanced_accuracy=0.75,
        counterfactual_reproduction=1.0,
        lookahead_consistent=True,
        independent_review=True,
        gates=gates,
    )
    verdict = record.verdict.model_copy(
        update={"status": status, "metrics": metrics, "policy": ReviewPolicy()}
    )
    wiki = tmp_path / "wiki"
    write_experiment_page(record.model_copy(update={"verdict": verdict}), wiki)
    return wiki


def _text(plan, **kwargs) -> str:
    return render(steps(plan, **kwargs))


def test_explore_steps_claim_the_unit_the_seed_picks(tmp_path, monkeypatch):
    monkeypatch.setattr(draw_steps, "model_dims", lambda model: (24, 4864))
    plan = plan_next(tmp_path / "wiki", seed=1, now=NOW)  # empty registry: explore
    seed = int(plan.choice.audit.split("--seed ")[1].split()[0])
    key = random_neuron(plan.choice.model, seed, (24, 4864)).replace("/", "-")
    text = _text(plan)
    assert text.startswith("1. Claim it before any other work")
    assert f"git switch -c experiment/{key}\n" in text
    assert f"--evidence-out exchange/{key}/ev.json" in text
    assert f"--out submissions/{key}.json" in text
    assert "openinterp show" not in text  # nothing to read for a new unit
    assert "wiki/draws/" in text  # a random draw the scan stops on is a result


def test_explore_steps_without_the_models_config(tmp_path, monkeypatch):
    monkeypatch.setattr(draw_steps, "model_dims", lambda model: None)
    text = _text(plan_next(tmp_path / "wiki", seed=1, now=NOW))
    assert "experiment/<KEY>" in text and "`Unit:` line" in text


def test_revise_steps_read_the_history_and_supersede(tmp_path, record):
    wiki = _registry(tmp_path, record, "NEEDS_REVISION")
    plan = next(
        p
        for p in (plan_next(wiki, seed=s, epsilon=0.0, now=NOW) for s in range(50))
        if p.choice.kind == "revise"
    )
    text = _text(plan, corpus="c.jsonl")
    key = "Qwen--Qwen2.5-0.5B-L3-N117-positive"
    assert f"experiment/{key}-revise" in text
    assert f"openinterp show {wiki}/experiments/Qwen--Qwen2.5-0.5B/L3/N117-positive\n" in text
    scan, submit = [line for line in text.splitlines() if "openinterp audit" in line]
    assert "--supersedes" not in scan and "--corpus c.jsonl" in scan
    assert "--supersedes" in submit  # with the hypothesis, not the scan
    assert "--explanation-scores" in text


def test_rescan_steps_scan_the_corpus_and_supersede(tmp_path, record):
    wiki = tmp_path / "wiki"
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus=BUILTIN_CORPUS,
        n_snippets=1,
        tokens_scanned=552,
        selection="random",
        harness_version="t",
    )
    sub = record.submission.model_copy(update={"provenance": provenance})
    write_experiment_page(record.model_copy(update={"submission": sub}), wiki)
    plan = next(
        p
        for p in (plan_next(wiki, seed=s, epsilon=0.0, now=NOW) for s in range(50))
        if p.choice.kind == "rescan"
    )
    text = _text(plan, corpus="c.jsonl")
    assert "experiment/Qwen--Qwen2.5-0.5B-L3-N117-positive-rescan" in text
    assert "openinterp show " in text
    scan, submit = [line for line in text.splitlines() if "openinterp audit" in line]
    assert "--supersedes" not in scan and "--corpus c.jsonl" in scan
    assert f"--supersedes {plan.choice.record}" in submit
    assert 'git commit -m "Rescan ' in text


def test_review_steps_with_api_models(tmp_path, record):
    wiki = _registry(tmp_path, record, "VERIFIED")
    plan = next(
        p
        for p in (plan_next(wiki, seed=s, epsilon=0.0, now=NOW) for s in range(50))
        if p.choice.kind == "replicate"
    )
    text = _text(plan, reviewer="openai/gpt-test")
    assert f"openinterp verify --audit-file {plan.choice.record}" in text
    assert "--agent-model openai/gpt-test" in text
    assert "neuron-reviewer" not in text and "openinterp audit" not in text
    assert "must differ from the record's proposer" in text


def test_cli_next_steps(tmp_path, record):
    from click.testing import CliRunner

    from openinterp.cli import cli

    wiki = _registry(tmp_path, record, "NEEDS_REVISION")
    args = ["next", "--wiki-dir", str(wiki), "--no-claims", "--epsilon", "0", "--steps"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Steps:\n1. Claim it before any other work" in result.output
    assert "Post the result, whatever it is" in result.output
