"""The two-step CLI flow: agents answer prompt files, so no API key or waiting process."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft
from openinterp.agents.reviewer import AdversarialProbe, ProbeSet
from openinterp.cli import cli
from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission, ScanEvidence
from tests.test_agents import TEXTS


@pytest.fixture
def flow(tmp_path, monkeypatch, toy_runner):
    toy_runner.model.config = SimpleNamespace()
    loads = []

    def load(model_name, device, dtype):
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


def test_proposer_and_reviewer_answer_files(flow):
    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    prompt = (flow.path / "ev.prompt.md").read_text()
    assert "ev.hypothesis.json" in prompt and '"counterfactuals"' in prompt  # schema included
    assert not (flow.path / "submissions").exists()  # scan only

    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    run("audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
        "--agent-model", "external/proposer", "--out", "sub.json")  # fmt: skip
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.submitting_agent_model == "external/proposer"
    assert sub.identifier == evidence.identifier and sub.top_examples == evidence.top_examples
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)  # measured

    loads = len(flow.loads)
    run("review", "--audit-file", "sub.json", "--emit-prompt", "review.prompt.md")
    assert len(flow.loads) == loads  # the reviewer's prompt never touches the model
    assert "review.probes.json" in (flow.path / "review.prompt.md").read_text()
    assert "review.prompt.probes.json" not in (flow.path / "review.prompt.md").read_text()

    probes = ProbeSet(
        probes=[
            AdversarialProbe(text=t, expected_active=i % 3 == 0, rationale="r")
            for i, t in enumerate(TEXTS * 2)
        ]
    )
    (flow.path / "review.probes.json").write_text(probes.model_dump_json())
    run("review", "--audit-file", "sub.json", "--probes", "review.probes.json",
        "--agent-model", "external/reviewer", "--wiki-dir", "wiki")  # fmt: skip
    (record_path,) = (flow.path / "wiki" / "experiments").rglob("*.json")
    record = AuditRecord.model_validate_json(record_path.read_text())
    assert record.verdict.reviewer_agent_model == "external/reviewer"
    assert [r.prompt for r in record.verdict.adversarial_test_results][: len(probes.probes)] == [
        p.text for p in probes.probes
    ]


@pytest.mark.parametrize(
    "args, message",
    [
        (["audit", "--evidence", "x.json"], "go together"),
        (["audit", "--evidence", "x.json", "--hypothesis", "x.json"], "--agent-model is required"),
        (["review", "--audit-file", "x.json", "--probes", "x.json"], "--agent-model is required"),
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
