"""Superpositions: sparse directions over one layer's neurons (issue #3215)."""

import pytest
import torch
from pydantic import ValidationError

from openinterp.core.profiles import SuperpositionMember, SuperpositionProfile, superposition_id
from openinterp.core.schemas import SuperpositionIdentifier
from openinterp.harness.superpositions import NULLS, directions, scan_superposition
from openinterp.harness.units import unit_of
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)

MEMBERS = [(3, 2.0), (7, -1.0), (20, 0.5)]


def _profile(members=MEMBERS):
    return SuperpositionProfile(
        members=[SuperpositionMember(neuron_idx=n, weight=w) for n, w in members],
        method="circuit",
        source="test",
    )


def _ident(profile, layer=1):
    return SuperpositionIdentifier(
        model_name="tiny", layer=layer, superposition_id=superposition_id(profile.members)
    )


def test_a_profile_needs_two_sorted_distinct_members_and_has_a_stable_id():
    with pytest.raises(ValidationError, match="at least 2"):
        _profile([(3, 1.0)])
    with pytest.raises(ValidationError, match="sorted"):
        _profile([(7, 1.0), (3, 1.0)])
    with pytest.raises(ValidationError, match="not be zero"):
        _profile([(3, 1.0), (7, 0.0)])
    p = _profile()
    scaled = _profile([(n, 4 * w) for n, w in MEMBERS])
    assert superposition_id(p.members) == superposition_id(scaled.members)
    assert superposition_id(p.members) != superposition_id(_profile([(3, 2.0), (7, 1.0)]).members)
    assert _ident(p).slug == f"tiny/L1/S{superposition_id(p.members)}"


def test_the_directions_are_the_unit_vector_then_matched_nulls():
    p = _profile()
    q = directions(_ident(p), p, width=48)
    assert q.shape == (48, 1 + NULLS)
    w = torch.zeros(48)
    for n, weight in MEMBERS:
        w[n] = weight
    assert torch.allclose(q[:, 0], w / w.norm())
    assert torch.allclose(q.norm(dim=0), torch.ones(1 + NULLS))
    assert all(int((q[:, j] != 0).sum()) == 3 for j in range(1 + NULLS))  # same support size
    assert torch.equal(q, directions(_ident(p), p, width=48))  # seeded by the id


def test_strength_is_the_positive_projection_and_ablation_removes_it(qwen_runner):  # noqa: F811
    p = _profile()
    unit = unit_of(_ident(p), superposition_profile=p)
    text = "abcabcab"
    _, cache = qwen_runner.run_with_cache(text, [1])
    w = directions(_ident(p), p, width=qwen_runner.mlp_width(1))[:, 0]
    expected = (cache[1][0].float() @ w).clamp(min=0)
    assert torch.allclose(unit.strengths(qwen_runner, text)[0], expected, atol=1e-5)

    # Removing the latent's write equals editing the hidden activations directly.
    def edit(act):
        return act - (act.float() @ w.to(act.device)).clamp(min=0)[..., None] * w.to(act.device)

    ablated = qwen_runner.run_with_interventions(text, **unit.ablation(qwen_runner))
    edited = qwen_runner.run_with_interventions(text, interventions={1: edit})
    assert torch.allclose(ablated, edited, atol=1e-4)
    assert not torch.allclose(ablated, qwen_runner.run_with_interventions(text), atol=1e-4)


def test_a_scan_reads_the_superposition_and_keeps_its_profile(qwen_runner):  # noqa: F811
    p = _profile()
    texts = ["abcabc", "bcabca", "cabcab", "aabbcc"]
    scan = scan_superposition(qwen_runner, texts, _ident(p), p, k=3)
    assert scan.profile == p
    assert scan.examples and scan.examples[0].max_activation >= scan.examples[-1].max_activation


TEXTS = ["abcabcabc", "cabbacabc", "aaabbbccc", "abcdabcd", "bcabcabca", "ccbbaaccb"]


def test_cli_audits_a_superposition_from_its_file(qwen_runner, tmp_path, monkeypatch):  # noqa: F811
    import json

    from click.testing import CliRunner

    from openinterp.cli import cli
    from openinterp.core.schemas import ScanEvidence

    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: qwen_runner)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    p = _profile()
    definition = {"model": "tiny", "layer": 1, **p.model_dump(mode="json")}
    (tmp_path / "s.json").write_text(json.dumps(definition))
    args = [
        "audit",
        "--superposition",
        "s.json",
        "--corpus",
        "corpus.txt",
        "--evidence-out",
        "ev.json",
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    evidence = ScanEvidence.model_validate_json((tmp_path / "ev.json").read_text())
    assert evidence.identifier == _ident(p) and evidence.superposition_profile == p
    assert evidence.provenance.selection == "manual"
    assert evidence.vocabulary_lens is not None and evidence.input_attribution is None
    prompt = (tmp_path / "ev.prompt.md").read_text()
    assert f"Superposition: tiny L1/S{_ident(p).superposition_id}" in prompt
    assert "one direction over 3 of the layer's neurons" in prompt
    bad = CliRunner().invoke(cli, [*args[:3], "--unit", "head", *args[3:]])
    assert bad.exit_code != 0 and "a unit of its own" in bad.output


def _reviewed(runner):
    """A reviewed record of the test superposition, measured on ``runner``."""
    from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft, ProposerAgent
    from openinterp.agents.reviewer import AdversarialProbe, evaluate_submission
    from openinterp.core.schemas import AuditRecord
    from openinterp.harness.hooks import activation_histogram
    from tests.test_agents import SMALL

    p = _profile()
    ident = _ident(p)
    scan = scan_superposition(runner, TEXTS, ident, p, k=4)
    pairs = [("abcabc", "abcxyz"), ("aaaa", "abcd"), ("cabc", "cxyz")]
    draft = HypothesisDraft(
        explanation="Fires on repeated letters.",
        activating_tokens=["a"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.3, rationale="r"
            )
            for b, c in pairs
        ],
    )
    sub = ProposerAgent("openai/proposer").build_submission(
        ident, draft, runner=runner, examples=scan.examples,
        histogram=activation_histogram(scan.values), confounds=scan.confounds,
        superposition_profile=p,
    )  # fmt: skip
    probes = [AdversarialProbe(text=t, expected_active=i % 2 == 0, rationale="r")
              for i, t in enumerate([*TEXTS, "xyzxyz", "plain", "abab", "nope!"])]  # fmt: skip
    verdict = evaluate_submission(sub, runner, probes, "openai/reviewer", SMALL)
    return AuditRecord(submission=sub, verdict=verdict)


def test_a_superposition_is_reviewed_and_reproduced(qwen_runner):  # noqa: F811
    from openinterp.core.schemas import AuditRecord
    from openinterp.reproduce import reproduce_record

    record = _reviewed(qwen_runner)
    sub, verdict = record.submission, record.verdict
    assert sub.superposition_profile == _profile() and sub.dictionary_profile is None
    assert verdict.causal_metrics is not None  # ranked against the null directions
    assert AuditRecord.model_validate_json(record.model_dump_json()) == record
    report = reproduce_record(record, qwen_runner)
    assert report.ok, report.mismatches


def test_a_superposition_record_has_a_wiki_page_and_passes_the_check(
    qwen_runner,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.checks import check_registry

    record = _reviewed(qwen_runner)
    monkeypatch.chdir(tmp_path)
    write_experiment_page(record, tmp_path / "wiki")
    (page,) = (tmp_path / "wiki").rglob("2*.md")
    slug = record.submission.identifier.slug
    assert page.parent.relative_to(tmp_path / "wiki" / "experiments").as_posix() == slug
    report = check_registry(tmp_path / "wiki", legacy_audits_dir=tmp_path / "audits")
    assert report.ok, [str(x) for x in report.problems]
