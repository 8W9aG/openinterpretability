"""The two-step CLI flow: agents answer prompt files, so no API key or waiting process."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp.agents.judge import ScoringTask
from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft
from openinterp.agents.reviewer import AdversarialProbe, ProbeSet
from openinterp.cli import cli
from openinterp.cli_costs import read_ledger
from openinterp.core.costs import LEDGER_ENV
from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission, ScanEvidence
from tests.test_agents import TEXTS


@pytest.fixture
def flow(tmp_path, monkeypatch, toy_runner):
    toy_runner.model.config = SimpleNamespace()
    loads = []

    def load(model_name, device, dtype, *_args, **_kwargs):
        loads.append(model_name)
        return toy_runner

    monkeypatch.setattr("openinterp.cli._load_runner", load)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    draft = HypothesisDraft(
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
    return SimpleNamespace(path=tmp_path, loads=loads, draft=draft)


def run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_audit_scans_windows_and_records_them(flow):
    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--window-tokens", 4, "--evidence-out", "ev.json")  # fmt: skip
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    prov = evidence.provenance
    assert (prov.snippet_tokens, prov.snippet_stride) == (4, 2)  # stride defaults to half
    assert prov.tokens_scanned == sum(len(t) for t in TEXTS)
    assert all(len(ex.tokens) <= 4 for ex in evidence.top_examples)

    args = ["audit", "--model", "toy", "--random", "--stride", "2", "--evidence-out", "ev.json"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0 and "--stride needs --window-tokens" in result.output


def test_proposer_and_reviewer_answer_files(flow, monkeypatch):
    monkeypatch.setenv(LEDGER_ENV, str(flow.path / "costs.jsonl"))
    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    prompt = (flow.path / "ev.prompt.md").read_text()
    assert "ev.hypothesis.json" in prompt and '"counterfactuals"' in prompt  # schema included
    # The toy model has no attention: every peak comes from its own token.
    assert evidence.input_attribution.current_token_share == pytest.approx(1.0)
    assert "Input attribution" in prompt and "100% of it is on the peak token" in prompt
    assert not (flow.path / "submissions").exists()  # scan only

    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    run("audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
        "--agent-model", "external/proposer", "--out", "sub.json")  # fmt: skip
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.submitting_agent_model == "external/proposer"
    assert sub.identifier == evidence.identifier and sub.top_examples == evidence.top_examples
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)  # measured
    assert sub.input_attribution == evidence.input_attribution

    loads = len(flow.loads)
    run("review", "--audit-file", "sub.json", "--emit-prompt", "review.prompt.md")
    assert len(flow.loads) == loads  # the reviewer's prompt never touches the model
    assert "review.probes.json" in (flow.path / "review.prompt.md").read_text()
    assert "review.prompt.probes.json" not in (flow.path / "review.prompt.md").read_text()
    assert "Probes are measured on their own" in (flow.path / "review.prompt.md").read_text()

    probes = ProbeSet(
        probes=[
            AdversarialProbe(text=t, expected_active=i % 3 == 0, rationale="r")
            for i, t in enumerate(TEXTS * 2)
        ]
    )
    (flow.path / "review.probes.json").write_text(probes.model_dump_json())
    (flow.path / "heldout.txt").write_text("\n".join(t[::-1] + " abc" for t in TEXTS))
    run("score", "--audit-file", "sub.json", "--corpus", "heldout.txt", "--window-tokens", 4,
        "--emit-task", "score.task.json")  # fmt: skip
    task = ScoringTask.model_validate_json((flow.path / "score.task.json").read_text())
    (flow.path / "score.detection.json").write_text('{"active": [1]}')
    (flow.path / "score.fuzzing.json").write_text('{"correct": []}')
    run("score", "--task", "score.task.json", "--judge-model", "external/judge",
        "--embedding-model", "")  # fmt: skip
    run("review", "--audit-file", "sub.json", "--probes", "review.probes.json",
        "--agent-model", "external/reviewer", "--explanation-scores", "score.scores.json",
        "--wiki-dir", "wiki")  # fmt: skip
    (record_path,) = (flow.path / "wiki" / "experiments").rglob("*.json")
    record = AuditRecord.model_validate_json(record_path.read_text())
    assert record.verdict.reviewer_agent_model == "external/reviewer"
    scores = record.verdict.explanation_scores
    assert scores.judge_model == "external/judge" and len(scores.detection_windows) == len(
        task.detection
    )
    assert [r.prompt for r in record.verdict.adversarial_test_results][: len(probes.probes)] == [
        p.text for p in probes.probes
    ]

    # Every step's costs reach the record: the scan's through the evidence file, the
    # scores' through the scores file, and the review's own.
    costs = record.submission.provenance.costs
    stages = [s.stage for s in costs.stages]
    judged = ["judge", "judge"] if task.fuzzing else ["judge"]
    assert stages == ["scan", "lenses", "input_attribution", "proposer", "counterfactuals",
                      "heldout_windows", *judged, "remeasure", "reviewer", "probe_measurement",
                      "gates", "causal_metrics"]  # fmt: skip
    assert [c.agent for c in costs.llm_calls] == ["proposer", *judged, "reviewer"]
    assert all(c.external and c.prompt_chars for c in costs.llm_calls)
    assert scores.costs is None  # moved into the provenance
    ledger = read_ledger([flow.path / "costs.jsonl"])
    assert [e["command"] for e in ledger] == [
        "audit",
        "audit",
        "review",
        "score",
        "score",
        "review",
    ]
    summary = run("costs", flow.path / "costs.jsonl").output
    assert "probe_measurement" in summary and "external: no token counts" in summary


def test_rereviewing_a_record_keeps_only_the_new_review_costs(flow, toy_runner):
    from openinterp.agents.reviewer import evaluate_submission

    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    run("audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
        "--agent-model", "external/proposer", "--out", "sub.json")  # fmt: skip
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert "scan" in [s.stage for s in sub.provenance.costs.stages]
    probes = ProbeSet(
        probes=[
            AdversarialProbe(text=t, expected_active=i % 3 == 0, rationale="r")
            for i, t in enumerate(TEXTS * 2)
        ]  # fmt: skip
    )
    old = AuditRecord(
        submission=sub, verdict=evaluate_submission(sub, toy_runner, probes.probes, "external/r")
    )
    (flow.path / "old.json").write_text(old.model_dump_json())
    (flow.path / "review.probes.json").write_text(probes.model_dump_json())
    (flow.path / "heldout.txt").write_text("\n".join(t[::-1] + " abc" for t in TEXTS))
    run("score", "--audit-file", "old.json", "--corpus", "heldout.txt", "--window-tokens", 4,
        "--emit-task", "score.task.json")  # fmt: skip
    (flow.path / "score.detection.json").write_text('{"active": [1]}')
    (flow.path / "score.fuzzing.json").write_text('{"correct": []}')
    run("score", "--task", "score.task.json", "--judge-model", "external/judge",
        "--embedding-model", "")  # fmt: skip
    run("review", "--audit-file", "old.json", "--probes", "review.probes.json",
        "--agent-model", "external/reviewer2", "--explanation-scores", "score.scores.json",
        "--wiki-dir", "", "--out", "new.json")  # fmt: skip
    new = AuditRecord.model_validate_json((flow.path / "new.json").read_text())
    stages = [s.stage for s in new.submission.provenance.costs.stages]
    assert stages[0] == "heldout_windows" and "probe_measurement" in stages
    assert not {"scan", "proposer"} & set(stages)  # the audit was another experiment's


@pytest.mark.parametrize(
    "args, message",
    [
        (["audit", "--evidence", "x.json"], "go together"),
        (["audit", "--evidence", "x.json", "--hypothesis", "x.json"], "--agent-model is required"),
        (["review", "--audit-file", "x.json", "--probes", "x.json"], "--agent-model is required"),
        (
            ["review", "--audit-file", "x.json", "--probes", "x.json", "--agent-model", "r"],
            "--explanation-scores is required",
        ),
    ],
)
def test_answer_files_need_a_recorded_model(flow, args, message):
    (flow.path / "x.json").write_text(json.dumps({"identifier": {}}))
    if args[0] == "review":
        submission = {
            "identifier": {"model_name": "toy", "layer": 1, "neuron_idx": 2, "pole": "positive"},
            "natural_language_explanation": "x",
            "activating_tokens": [],
            "counterfactuals": [
                {"base_prompt": "a", "counterfactual_prompt": "b", "expected_delta": 0.0}
            ],
            "lookahead": {"is_lookahead": False, "kl_divergence": 0.0},
            "submitting_agent_model": "external/proposer",
        }
        (flow.path / "x.json").write_text(json.dumps(submission))
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0 and message in result.output
