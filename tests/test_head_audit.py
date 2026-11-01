"""Auditing an attention head end to end, on a randomly initialised GPT-2 (no download)."""

import json

import pyarrow.parquet as pq
import pytest
import torch
from click.testing import CliRunner

from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft, ProposerAgent
from openinterp.agents.reviewer import (
    AdversarialProbe,
    ProbeSet,
    ReviewerAgent,
    evaluate_submission,
    repeated_runs,
)
from openinterp.builder.compiler import compile_site
from openinterp.builder.wiki import write_experiment_page
from openinterp.cli import cli
from openinterp.core.schemas import AuditRecord, HeadIdentifier, ScanEvidence
from openinterp.harness.heads import pattern_scores, scan_head
from openinterp.harness.hooks import NeuronHookRunner, activation_histogram
from openinterp.harness.units import HeadUnit, head_strength, submission_unit
from openinterp.planner import plan_next
from openinterp.reproduce import reproduce_record
from tests.conftest import FakeTokenizer
from tests.test_agents import SMALL

HEAD = HeadIdentifier(model_name="tiny-gpt2", layer=1, head_idx=1)
TEXTS = ["abcabcabc", "hello hello", "zyxw", "quoted 'text'", "mixed Case 123", "aaaa bbbb"]


@pytest.fixture
def runner(tiny_gpt2):
    tiny_gpt2.set_attn_implementation("eager")
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def scan(runner):
    return scan_head(runner, TEXTS, HEAD, k=4, pattern_sequences=2, pattern_length=6)


def test_scan_measures_strength_from_the_mean_output(runner, scan):
    z = torch.cat([runner.head_outputs(t, 1)[0, :, 1] for t in TEXTS])
    profile = scan.profile
    torch.testing.assert_close(torch.tensor(profile.mean_output), z.mean(0), atol=1e-5, rtol=0)
    # z̄ is over every counted token; with a first-token floor the values leave firsts out.
    assert profile.mean_tokens == sum(map(len, TEXTS))
    assert len(scan.values) == profile.mean_tokens - len(TEXTS) * scan.first_token_floor
    assert scan.examples[0].max_activation == pytest.approx(max(scan.values))
    unit = submission_unit(scan_evidence(scan))
    for ex in scan.examples:
        measured = unit.strengths(runner, ex.text)[0]
        torch.testing.assert_close(measured, torch.tensor(ex.activations), atol=1e-4, rtol=0)
        assert ex.attends_to and max(ex.attends_to) <= ex.max_token_index
        assert all(0 < w <= 1 for w in ex.attention_weights)


def test_mean_ablation_zeroes_the_head_and_nothing_else(runner, scan):
    mean = tuple(scan.profile.mean_output)
    ablate = HeadUnit(HEAD, mean).ablation(runner)["head_interventions"][HEAD.layer]
    z = runner.head_outputs(TEXTS[0], 1).flatten(-2)
    ablated = ablate(z).reshape(*z.shape[:-1], 2, 16)
    torch.testing.assert_close(
        head_strength(runner, HEAD, ablated[..., 1, :], mean), torch.zeros(1, 9)
    )
    torch.testing.assert_close(ablated[..., 0, :], z.reshape(*z.shape[:-1], 2, 16)[..., 0, :])


def test_pattern_scores_read_a_perfect_induction_head(runner, monkeypatch):
    length = 5

    def induction(inputs, layer):
        att = torch.zeros(inputs.shape[0], 2, 2 * length, 2 * length)
        for q in range(2 * length):
            att[:, 1, q, q - length + 1 if q >= length else 0] = 1.0
        return att

    monkeypatch.setattr(runner, "attention_patterns", induction)
    scores = pattern_scores(runner, 0, 1, sequences=3, length=length, seed=0)
    assert (scores["induction_score"], scores["duplicate_token_score"]) == (1.0, 0.0)


def scan_evidence(scan):
    return ScanEvidence(
        identifier=HEAD,
        top_examples=scan.examples,
        confounds=scan.confounds,
        head_profile=scan.profile,
        model_n_heads=2,
    )


def _record(runner, scan):
    draft = HypothesisDraft(
        explanation="Fires on the second copy of a repeated letter sequence.",
        activating_tokens=["c"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.3, rationale="r"
            )
            for b, c in [("abcabc", "abcxyz"), ("hello hello", "hello world"), ("zz", "zy")]
        ],
    )
    sub = ProposerAgent("openai/proposer").build_submission(
        HEAD,
        draft,
        runner=runner,
        examples=scan.examples,
        histogram=activation_histogram(scan.values),
        confounds=scan.confounds,
        head_profile=scan.profile,
        first_token_floor=scan.first_token_floor,
    )
    probes = [
        AdversarialProbe(text=t, expected_active=i % 2 == 0, rationale="r")
        for i, t in enumerate(TEXTS + ["xyzxyz", "plain", "abab", "nope!"])
    ]
    # The toy head is default-on (see below); these tests are about everything else.
    policy = SMALL.model_copy(update={"max_active_share": 1.0})
    verdict = evaluate_submission(sub, runner, probes, "openai/reviewer", policy)
    return AuditRecord(submission=sub, verdict=verdict)


def test_a_head_is_audited_reviewed_and_reproduced(runner, scan):
    record = _record(runner, scan)
    sub, verdict = record.submission, record.verdict
    assert (sub.model_n_heads, sub.correlated_neurons, sub.vocabulary_lens) == (2, [], None)
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)
    assert verdict.joint_ablation is None  # partners are neuron poles
    if verdict.causal_metrics is not None:  # ranked against the layer's other head
        assert verdict.causal_metrics.null_neurons == 1
    assert AuditRecord.model_validate_json(record.model_dump_json()) == record
    report = reproduce_record(record, runner)
    assert report.ok, report.mismatches

    prompt = ReviewerAgent.probe_prompt(sub)
    assert prompt.startswith("Attention head: tiny-gpt2 L1/H1") and "residual stream" in prompt
    assert "reads tokens, not words" in prompt


def test_repeated_runs_are_maximal_token_matches():
    # "x a b y a b a": "a b" repeats at 4-5, then "a" alone at 6 ("b a" is new).
    assert repeated_runs([7, 1, 2, 8, 1, 2, 1]) == [(4, 6), (6, 7)]
    assert repeated_runs([1, 2, 3]) == []
    assert repeated_runs([5, 5, 5]) == [(1, 3)]  # overlapping self-repeat


def test_review_check_probes_shows_tokens_and_repeats(runner, scan, tmp_path, monkeypatch):
    sub = _record(runner, scan).submission
    (tmp_path / "sub.json").write_text(sub.model_dump_json())
    probes = ProbeSet(
        probes=[
            AdversarialProbe(text=t, expected_active=i < 4, rationale="r")
            for i, t in enumerate(["abab", "xyz", "abcd", "qq", "mn", "op", "rs", "tu", "vw", "jk"])
        ]
    )
    (tmp_path / "review.probes.json").write_text(probes.model_dump_json())
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a: pytest.fail("loaded a model"))
    args = ["review", "--audit-file", str(tmp_path / "sub.json"),
            "--check-probes", str(tmp_path / "review.probes.json")]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    prompt = (tmp_path / "review.check.prompt.md").read_text()
    assert "reads tokens, not words" in prompt and "review.probes.json" in prompt
    tok = FakeTokenizer()
    ab = "".join(tok.decode([i]) for i in tok.encode("ab"))
    assert "0. expected active: 'abab'" in prompt
    assert f"repeated token runs: {ab!r} (2)" in prompt
    assert "1. expected active: 'xyz'" in prompt and "repeated token runs: none" in prompt


def test_head_pages_parquet_and_follow_ups(runner, scan, tmp_path):
    record = _record(runner, scan)
    wiki = tmp_path / "wiki"
    page = write_experiment_page(record, wiki)
    assert page.parent == wiki / "experiments" / "tiny-gpt2" / "L1" / "H1"
    assert "# tiny-gpt2 · L1/H1 ·" in page.read_text()
    assert "L1/H1" in (wiki / "README.md").read_text()

    report = compile_site([record], tmp_path / "dist")
    html = (tmp_path / "dist" / "models" / "tiny-gpt2" / "L1" / "H1.html").read_text()
    assert "attention head" in html and "Induction" in html and "Peak attends to" in html
    (row,) = pq.read_table(report.parquet).to_pylist()
    assert (row["kind"], row["head_idx"], row["neuron_idx"], row["pole"]) == (
        "attn_head",
        1,
        None,
        None,
    )
    assert row["induction_score"] == scan.profile.induction_score
    assert "Heads studied" in (tmp_path / "dist" / "index.html").read_text()

    plan = plan_next(wiki, epsilon=0.0, seed=0)
    kinds = {c.kind for c in plan.candidates}
    assert "opposite_pole" not in kinds
    audits = [cmd for c in plan.candidates for cmd in c.commands if "openinterp audit" in cmd]
    assert audits and all("--head 1" in cmd and "--pole" not in cmd for cmd in audits)


def test_cli_samples_a_random_head_with_eager_attention(runner, tmp_path, monkeypatch):
    loads = []

    def load(model_name, device, dtype, revision=None, attn=None):
        loads.append(attn)
        return runner

    monkeypatch.setattr("openinterp.cli._load_runner", load)
    # A randomly initialised head writes far from its mean on most tokens: default-on.
    monkeypatch.setattr("openinterp.evidence.check_default_on", lambda *a: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    args = ["audit", "--model", "tiny-gpt2", "--unit", "head", "--random", "--seed", "3",
            "--corpus", "corpus.txt", "--evidence-out", "ev.json"]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    evidence = ScanEvidence.model_validate_json((tmp_path / "ev.json").read_text())
    assert evidence.identifier.kind == "attn_head" and evidence.head_profile is not None
    assert evidence.model_n_heads == 2 and evidence.vocabulary_lens is None
    assert loads == ["eager"]
    prompt = (tmp_path / "ev.prompt.md").read_text()
    assert "Attention head:" in prompt and "peak attends to" in prompt and "induction" in prompt

    bad = CliRunner().invoke(cli, ["audit", "--model", "m", "--head", "1", "--pole", "positive"])
    assert bad.exit_code != 0 and "a head has neither" in bad.output
    assert json.loads((tmp_path / "ev.json").read_text())["identifier"]["kind"] == "attn_head"


def test_head_scans_take_label_sets(runner):
    from openinterp.harness.annotations import DocumentLabels

    langs = ["en", "fr"] * 3
    tagged = scan_head(
        runner, TEXTS, HEAD, k=4, pattern_sequences=2, pattern_length=6,
        taggers=[DocumentLabels("language", langs)],
    )  # fmt: skip
    language = tagged.confounds.labels["language"]
    assert language.labelled_tokens == tagged.profile.mean_tokens
    assert {s.label for s in language.levels} == {"en", "fr"}
    for ex in tagged.examples:
        assert ex.peak_labels == {"language": langs[TEXTS.index(ex.text)]}
