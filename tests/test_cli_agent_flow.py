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
from openinterp.core.contributors import CONTRIBUTOR_ENV
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
    monkeypatch.delenv(CONTRIBUTOR_ENV, raising=False)
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


def test_audit_warns_about_a_corpus_without_a_matching_sidecar(flow):
    from openinterp.corpus import write_corpus

    args = ["audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--evidence-out", "ev.json"]
    assert "has no corpus.source.json" in run(*args, "--corpus", "corpus.txt").output
    source = write_corpus(TEXTS, flow.path / "built.jsonl", dataset="d/x", files="*",
                          seed=0, max_chars=99, docs_per_row_group=1)  # fmt: skip
    assert "warning" not in run(*args, "--corpus", "built.jsonl").output
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    assert evidence.provenance.corpus_source == source


def test_audit_without_corpus_scans_corpus_jsonl_or_fails(flow):
    args = ["audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--evidence-out", "ev.json"]
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code != 0 and "No --corpus given" in result.output
    assert not (flow.path / "ev.json").exists()  # never the built-in sample

    (flow.path / "corpus.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    assert "Scanning corpus.jsonl" in run(*args).output
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    assert evidence.provenance.corpus == "corpus.jsonl"


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


def test_audit_scans_a_seeded_token_sample(flow):
    def scan(*args):
        run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
            *args, "--evidence-out", "ev.json")  # fmt: skip
        return ScanEvidence.model_validate_json((flow.path / "ev.json").read_text()).provenance

    prov = scan("--max-tokens", 30, "--sample-seed", 5)
    sample = prov.corpus_sample
    assert (sample.max_tokens, sample.seed, sample.corpus_documents) == (30, 5, len(TEXTS))
    assert prov.n_snippets == sample.documents < len(TEXTS)
    assert prov.tokens_scanned == sample.tokens <= 30  # one token per character
    assert scan("--max-tokens", 30, "--sample-seed", 5).corpus_sample == sample
    assert scan().corpus_sample.documents == len(TEXTS)  # the default budget fits it all
    assert scan("--max-tokens", 0).corpus_sample is None  # the whole corpus


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
        "--agent-model", "external/proposer", "--contributor", "alice",
        "--out", "sub.json")  # fmt: skip
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.submitting_agent_model == "external/proposer" and sub.contributor == "alice"
    assert sub.identifier == evidence.identifier and sub.top_examples == evidence.top_examples
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)  # measured
    assert sub.input_attribution == evidence.input_attribution
    # The scan read the other pole too, for the neuron page.
    assert sub.opposite_pole == evidence.opposite_pole and sub.opposite_pole.pole == "negative"
    # The Patchscope reads the direction only after the proposer answered.
    assert sub.patchscope is not None and sub.patchscope.readings
    assert "patchscope" not in (flow.path / "ev.json").read_text()
    # So are the steering samples, from neutral prompts no agent sees.
    assert sub.steering is not None and len(sub.steering.samples) == 5
    assert "steering" not in (flow.path / "ev.json").read_text()
    # And the wiring from the weights (the toy model has no attention, so no heads).
    assert sub.wiring.method == "weights" and sub.wiring.upstream_heads is None
    assert "wiring" not in (flow.path / "ev.json").read_text()

    loads = len(flow.loads)
    run("review", "--audit-file", "sub.json", "--emit-prompt", "review.prompt.md")
    assert len(flow.loads) == loads  # the reviewer's prompt never touches the model
    assert "review.probes.json" in (flow.path / "review.prompt.md").read_text()
    assert "review.prompt.probes.json" not in (flow.path / "review.prompt.md").read_text()
    assert "Probes are measured on their own" in (flow.path / "review.prompt.md").read_text()
    assert sub.patchscope.readings[0].text not in (flow.path / "review.prompt.md").read_text()
    assert sub.steering.samples[0].prompt not in (flow.path / "review.prompt.md").read_text()

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
    monkeypatch.setenv(CONTRIBUTOR_ENV, "bob")
    run("review", "--audit-file", "sub.json", "--probes", "review.probes.json",
        "--agent-model", "external/reviewer", "--explanation-scores", "score.scores.json",
        "--wiki-dir", "wiki")  # fmt: skip
    (record_path,) = (flow.path / "wiki" / "experiments").rglob("*.json")
    record = AuditRecord.model_validate_json(record_path.read_text())
    assert record.verdict.reviewer_agent_model == "external/reviewer"
    assert (record.submission.contributor, record.verdict.contributor) == ("alice", "bob")
    page = record_path.with_suffix(".md").read_text()
    assert 'reviewer_contributor: "bob"' in page and "`external/proposer`, run by alice" in page
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
                      "patchscope", "steering", "wiring", "heldout_windows", *judged,
                      "remeasure", "reviewer", "probe_measurement", "gates", "causal_metrics",
                      "wiring"]  # fmt: skip
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
    assert (
        "probe_measurement" in summary
        and "external without a usage file: no token counts" in summary
    )


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


def test_contributor_is_optional_and_never_an_email(flow, monkeypatch):
    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    step = ["audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
            "--agent-model", "external/proposer", "--out", "sub.json"]  # fmt: skip
    run(*step)
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.contributor is None  # nothing given: nobody recorded, git config unread
    monkeypatch.setenv(CONTRIBUTOR_ENV, "someone@example.com")
    result = CliRunner().invoke(cli, step)
    assert result.exit_code != 0 and "never an email" in result.output


def test_model_aliases_rename_the_recorded_answerer(flow, monkeypatch):
    # e.g. Claude Code proxied to Grok: the subagent reports the Claude model it was asked for.
    monkeypatch.setenv("OPENINTERP_MODEL_ALIASES", "claude-opus-5-5=grok-4")
    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    run("audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
        "--agent-model", "external/claude-opus-5-5", "--out", "sub.json")  # fmt: skip
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.submitting_agent_model == "external/grok-4"

    monkeypatch.setenv("OPENINTERP_MODEL_ALIASES", "claude-opus-5-5")
    result = CliRunner().invoke(cli, ["review", "--audit-file", "sub.json", "--agent-model",
                                      "external/r", "--emit-prompt", "x.md"])  # fmt: skip
    assert result.exit_code == 0  # a bare name is a default, not malformed
    monkeypatch.setenv("OPENINTERP_MODEL_ALIASES", "=grok-4")
    result = CliRunner().invoke(cli, ["review", "--audit-file", "sub.json", "--agent-model",
                                      "external/r", "--emit-prompt", "x.md"])  # fmt: skip
    assert result.exit_code != 0 and "malformed entry" in result.output


def test_audit_records_the_experiment_it_supersedes(flow, record):
    from openinterp.builder.wiki import archive_record

    run("audit", "--model", "toy", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--evidence-out", "ev.json")  # fmt: skip
    (flow.path / "ev.hypothesis.json").write_text(flow.draft.model_dump_json())
    evidence = ScanEvidence.model_validate_json((flow.path / "ev.json").read_text())
    same = record.model_copy(
        update={
            "submission": record.submission.model_copy(update={"identifier": evidence.identifier})
        }
    )
    old = archive_record(same, flow.path / "wiki")
    step = ["audit", "--evidence", "ev.json", "--hypothesis", "ev.hypothesis.json",
            "--agent-model", "external/proposer", "--out", "sub.json"]  # fmt: skip
    result = run(*step, "--supersedes", old)
    assert "Supersedes: 20261001T120000Z (VERIFIED)" in result.output
    sub = NeuronAuditSubmission.model_validate_json((flow.path / "sub.json").read_text())
    assert sub.supersedes == "20261001T120000Z"

    other = archive_record(record, flow.path / "other")  # another unit
    result = CliRunner().invoke(cli, [*step, "--supersedes", str(other)])
    assert result.exit_code != 0 and "names an experiment on" in result.output
    result = CliRunner().invoke(cli, ["audit", "--model", "toy", "--random", "--evidence-out",
                                      "ev.json", "--supersedes", str(old)])  # fmt: skip
    assert result.exit_code != 0 and "--supersedes goes on the step" in result.output


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


def test_a_random_draw_the_scan_stops_on_is_recorded(flow):
    from openinterp.checks import check_registry
    from openinterp.core.draws import load_draws

    # The toy pole is active on most tokens: default-on.
    args = ["audit", "--model", "toy", "--layer", "1", "--neuron", "7", "--pole", "positive",
            "--corpus", "corpus.txt", "--evidence-out", "ev.json"]  # fmt: skip
    manual = CliRunner().invoke(cli, args)
    assert manual.exit_code != 0 and "default-on" in manual.output
    assert not (flow.path / "wiki" / "draws").exists()  # only random draws are kept

    result = CliRunner().invoke(cli, [*args, "--random", "--seed", "4"])
    assert result.exit_code != 0 and "Recorded the default-on draw" in result.output
    (draw,) = load_draws(flow.path / "wiki")
    assert (draw.outcome, draw.identifier.pole, draw.provenance.seed) == (
        "default_on",
        "positive",
        4,
    )
    assert draw.provenance.selection == "random" and draw.active_share >= 0.5
    report = check_registry(flow.path / "wiki")
    assert report.ok and report.draws == 1
