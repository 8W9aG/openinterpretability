"""Reproducing a record catches edited numbers, results and verdicts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp.agents.reviewer import evaluate_submission
from openinterp.cli import cli
from openinterp.core.schemas import AuditRecord
from openinterp.reproduce import reproduce_record
from tests.test_agents import SMALL, _probes, _submission


@pytest.fixture
def draft():
    from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft

    return HypothesisDraft(
        explanation="Fires on repeated letters.",
        activating_tokens=["a"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.5, rationale="r"
            )
            for b, c in [("aaaa", "abcd"), ("zzzz", "zyxw"), ("bbbb", "bcde")]
        ],
    )


@pytest.fixture
def ident():
    from openinterp.core.schemas import NeuronIdentifier

    return NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")


@pytest.fixture
def record(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, "external/reviewer", SMALL)
    return AuditRecord(submission=sub, verdict=verdict)


def edit_result(record, index, **update):
    results = list(record.verdict.adversarial_test_results)
    results[index] = results[index].model_copy(update=update)
    verdict = record.verdict.model_copy(update={"adversarial_test_results": results})
    return record.model_copy(update={"verdict": verdict})


def test_honest_record_reproduces(record, toy_runner):
    report = reproduce_record(record, toy_runner)
    assert report.ok, report.mismatches
    assert report.compared > len(record.verdict.adversarial_test_results)
    assert record.verdict.environment.dtype == "float32"  # pinned at review time


def test_edited_probe_activation_is_caught(record, toy_runner):
    tampered = edit_result(
        record, 0, activation=record.verdict.adversarial_test_results[0].activation + 1.0
    )
    report = reproduce_record(tampered, toy_runner)
    assert not report.ok and "activation recorded" in report.mismatches[0]


def test_flipped_result_and_status_are_caught(record, toy_runner):
    first = record.verdict.adversarial_test_results[0]
    tampered = edit_result(record, 0, passed=not first.passed)
    tampered = tampered.model_copy(
        update={"verdict": tampered.verdict.model_copy(update={"status": "REJECTED"})}
    )
    messages = " ".join(reproduce_record(tampered, toy_runner).mismatches)
    assert "passed=" in messages and "status recorded REJECTED" in messages


def test_edited_scan_evidence_is_caught(record, toy_runner):
    sub = record.submission
    ex = sub.top_examples[0]
    louder = ex.model_copy(update={"activations": [a + 1.0 for a in ex.activations]})
    tampered = record.model_copy(
        update={
            "submission": sub.model_copy(
                update={"top_examples": [louder, *sub.top_examples[1:]], "max_activation": 99.0}
            )
        }
    )
    messages = " ".join(reproduce_record(tampered, toy_runner).mismatches)
    assert "top example 1: activations differ" in messages
    assert "is not the top example's peak" in messages


def test_other_protocol_versions_are_refused(record, toy_runner):
    old_policy = record.verdict.policy.model_copy(update={"protocol_version": "0.4"})
    old = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"policy": old_policy})}
    )
    (message,) = reproduce_record(old, toy_runner).mismatches
    assert "protocol 0.4" in message and "re-evaluate" in message


def test_cli_exit_code(tmp_path, monkeypatch, record, toy_runner):
    toy_runner.model.config = SimpleNamespace()
    loads = []
    monkeypatch.setattr(
        "openinterp.cli._load_runner",
        lambda name, device, dtype, revision=None: loads.append((name, dtype)) or toy_runner,
    )
    good = tmp_path / "good.json"
    good.write_text(record.model_dump_json())
    bad = tmp_path / "bad.json"
    bad.write_text(edit_result(record, 0, passed=False).model_dump_json())

    result = CliRunner().invoke(cli, ["reproduce", str(good)])
    assert result.exit_code == 0, result.output
    assert "REPRODUCED" in result.output and loads == [("toy", "float32")]

    result = CliRunner().invoke(cli, ["reproduce", str(good), str(bad)])
    assert result.exit_code == 1 and "FAILED" in result.output
    assert "1/2 record(s) reproduced" in result.output
    assert len(loads) == 2  # one load per invocation, shared across same-model records


def test_cli_allow_list_refuses_other_models_without_loading(tmp_path, monkeypatch, record):
    monkeypatch.setattr(
        "openinterp.cli._load_runner", lambda *a, **k: pytest.fail("must not load the model")
    )
    path = tmp_path / "r.json"
    path.write_text(record.model_dump_json())
    result = CliRunner().invoke(cli, ["reproduce", str(path), "--allow-model", "Qwen/Qwen2.5-0.5B"])
    assert result.exit_code == 1 and "model toy is not on the allow-list" in result.output


def test_cli_reports_to_github_actions(tmp_path, monkeypatch, record, toy_runner):
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bad = tmp_path / "bad.json"
    bad.write_text(edit_result(record, 0, passed=False).model_dump_json())
    result = CliRunner().invoke(cli, ["reproduce", str(bad)])
    assert result.exit_code == 1
    assert f"::error file={bad}::" in result.output
    assert "### Reproduction" in summary.read_text() and "❌" in summary.read_text()


def test_changed_records_lists_added_records(tmp_path, monkeypatch):
    import subprocess

    from openinterp.checks import changed_records

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    monkeypatch.chdir(tmp_path)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    old = tmp_path / "wiki" / "experiments" / "m" / "old.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("switch", "-q", "-c", "pr")
    (old.parent / "new.json").write_text("{}")
    (old.parent / "new.md").write_text("page")
    git("add", "-A")
    git("commit", "-q", "-m", "add")
    assert changed_records(Path("wiki"), "main") == [Path("wiki/experiments/m/new.json")]
