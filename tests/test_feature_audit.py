"""Auditing a sparse-dictionary latent end to end, on a randomly initialised GPT-2 and a
random transcoder stored locally (no download)."""

import pyarrow.parquet as pq
import pytest
import torch
from click.testing import CliRunner
from safetensors.torch import save_file

from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft, ProposerAgent
from openinterp.agents.reviewer import AdversarialProbe, ReviewerAgent, evaluate_submission
from openinterp.builder.compiler import compile_site
from openinterp.builder.wiki import write_experiment_page
from openinterp.cli import cli
from openinterp.core.schemas import AuditRecord, FeatureIdentifier, ScanEvidence
from openinterp.harness import dictionaries
from openinterp.harness.dictionaries import DictionarySpec, load_spec
from openinterp.harness.features import patch_features, scan_feature
from openinterp.harness.hooks import NeuronHookRunner, activation_histogram
from openinterp.harness.units import FeatureUnit
from openinterp.planner import plan_next
from openinterp.reproduce import reproduce_record
from tests.conftest import FakeTokenizer
from tests.test_agents import SMALL

FEATURE = FeatureIdentifier(model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=5)
TEXTS = ["abcabcabc", "hello hello", "zyxw", "quoted 'text'", "mixed Case 123", "aaaa bbbb"]
N_FEATURES = 48


@pytest.fixture
def spec(tmp_path, monkeypatch):
    torch.manual_seed(1)
    for layer in (0, 1):
        weights = {
            # stored the way some releases do, [n, d_in], to exercise the transpose on load
            "W_enc": torch.randn(N_FEATURES, 32) / 32**0.5,
            "b_enc": torch.zeros(N_FEATURES),
            "W_dec": torch.randn(N_FEATURES, 32) * 0.1,
            "b_dec": torch.zeros(32),
        }
        (tmp_path / f"tc{layer}").mkdir()
        save_file(weights, tmp_path / f"tc{layer}" / "sae.safetensors")
    spec = DictionarySpec(
        model_name="tiny-gpt2",
        kind="transcoder",
        repo=str(tmp_path),
        revision="local",
        path="tc{layer}/sae.safetensors",
        layers=(0, 1),
        input_site="mlp_in_normalized",
    )
    monkeypatch.setitem(dictionaries.DICTIONARIES, "tiny-tc", spec)
    return spec


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def scan(runner, spec):
    return scan_feature(runner, TEXTS, FEATURE, spec, k=4)


def _captured(runner, text, layer=1):
    """The ln_2 input, the MLP input and the MLP output of ``layer`` on ``text``."""
    seen = {}
    block = runner.model.transformer.h[layer]
    hooks = [
        block.ln_2.register_forward_pre_hook(lambda _m, a: seen.__setitem__("pre", a[0])),
        block.mlp.register_forward_pre_hook(lambda _m, a: seen.__setitem__("in", a[0])),
        block.mlp.register_forward_hook(lambda _m, _a, o: seen.__setitem__("out", o)),
    ]
    with torch.no_grad():
        runner.forward(runner.encode(text))
    for h in hooks:
        h.remove()
    return seen


def test_dictionary_sites_read_the_mlp_input_its_normalisation_and_output(runner):
    seen = _captured(runner, TEXTS[0])
    pre = seen["pre"]
    variance = pre.var(-1, correction=0, keepdim=True)
    standardised = (pre - pre.mean(-1, keepdim=True)) / (variance + 1e-5).sqrt()
    for site, expected in [
        ("mlp_in", seen["in"]),
        ("mlp_in_normalized", standardised),
        ("mlp_out", seen["out"]),
    ]:
        x, out = runner.dictionary_io(TEXTS[0], 1, site)
        torch.testing.assert_close(x.float(), expected.float(), atol=1e-5, rtol=0)
        torch.testing.assert_close(out, seen["out"])


def test_scan_measures_the_latent_and_the_dictionarys_fit(runner, spec, scan):
    unit = FeatureUnit(FEATURE, scan.profile)
    for ex in scan.examples:
        measured = unit.strengths(runner, ex.text)[0]
        torch.testing.assert_close(measured, torch.tensor(ex.activations), atol=1e-4, rtol=0)
    assert scan.examples[0].max_activation == pytest.approx(max(scan.values))

    dictionary = load_spec(spec, 1)
    xs, outs = zip(*(runner.dictionary_io(t, 1, spec.input_site) for t in TEXTS), strict=True)
    x, out = torch.cat([v[0] for v in xs]), torch.cat([v[0] for v in outs])
    acts = dictionary.encode(x)
    error = ((dictionary.decode(acts) - out) ** 2).sum()
    profile = scan.profile
    assert profile.explained_variance == pytest.approx(
        float(1 - error / ((out - out.mean(0)) ** 2).sum()), rel=1e-5
    )
    assert profile.mean_l0 == pytest.approx(float((acts > 0).sum(-1).float().mean()), abs=1e-3)
    assert profile.density == pytest.approx(float((acts[:, 5] > 0).float().mean()), abs=1e-5)
    assert (profile.n_features, profile.tokens, profile.path) == (
        N_FEATURES,
        len(scan.values),
        "tc1/sae.safetensors",
    )


def test_ablation_removes_exactly_the_latents_write(runner, scan):
    unit = FeatureUnit(FEATURE, scan.profile)
    dictionary = unit.dictionary(runner)
    x, clean = runner.dictionary_io(TEXTS[1], 1, "mlp_in_normalized")
    seen = []  # on the MLP block, so it sees the down projection's output after the ablation
    mlp = runner.model.transformer.h[1].mlp
    handle = mlp.register_forward_hook(lambda _m, _a, out: seen.append(out))
    runner.run_with_interventions(TEXTS[1], **unit.ablation(runner))
    handle.remove()
    write = dictionary.feature(x, 5).unsqueeze(-1) * dictionary.w_dec[5]
    assert float(write.abs().max()) > 0
    torch.testing.assert_close(seen[-1], clean - write, atol=1e-6, rtol=0)


def test_patching_a_latent_moves_the_output_by_its_write_change(runner, spec, scan):
    dictionary = load_spec(spec, 1)
    source, target = "abcabc", "abcxyz"
    site = spec.input_site
    a = {t: dictionary.encode(runner.dictionary_io(t, 1, site)[0][0]) for t in (source, target)}
    _, clean = runner.dictionary_io(target, 1, site)

    def mlp_change(_patched, _target):  # the effect: how far layer 1's MLP output moved
        return float(captured[-1][:1].sub(clean).abs().sum())

    # On the MLP block, so it sees the down projection's output after the patch.
    captured = []
    mlp = runner.model.transformer.h[1].mlp
    handle = mlp.register_forward_hook(lambda _m, _a, out: captured.append(out))
    features = [5, 7]
    effects = [
        patch_features(runner, dictionary, site, source, target, 1, [f], effect=mlp_change)[0]
        for f in features
    ]
    handle.remove()
    for f, effect in zip(features, effects, strict=True):
        expected = ((a[source][:, f] - a[target][:, f]).unsqueeze(-1) * dictionary.w_dec[f]).abs()
        assert effect == pytest.approx(float(expected.sum()), rel=1e-4)

    def logit_change(patched, unpatched):
        return float((patched - unpatched).abs().max())

    same = patch_features(
        runner, dictionary, site, target, target, 1, features, effect=logit_change
    )
    assert same == pytest.approx([0.0, 0.0], abs=1e-5)


def _record(runner, scan):
    draft = HypothesisDraft(
        explanation="Fires on the second copy of a repeated letter sequence.",
        activating_tokens=["c"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.3, rationale="r"
            )
            # minimal pairs that move this random latent
            for b, c in [("quoted 'text'", "qaoted 'text'"), ("zyxw", "zyxa"), ("zz", "zy")]
        ],
    )
    sub = ProposerAgent("openai/proposer").build_submission(
        FEATURE,
        draft,
        runner=runner,
        examples=scan.examples,
        histogram=activation_histogram(scan.values),
        confounds=scan.confounds,
        dictionary_profile=scan.profile,
    )
    probes = [
        AdversarialProbe(text=t, expected_active=i % 2 == 0, rationale="r")
        for i, t in enumerate(TEXTS + ["xyzxyz", "plain", "abab", "nope!"])
    ]
    verdict = evaluate_submission(sub, runner, probes, "openai/reviewer", SMALL)
    return AuditRecord(submission=sub, verdict=verdict)


def test_a_feature_is_audited_reviewed_and_reproduced(runner, scan):
    record = _record(runner, scan)
    sub, verdict = record.submission, record.verdict
    assert sub.dictionary_profile == scan.profile and sub.head_profile is None
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)
    assert verdict.joint_ablation is None  # partners are neuron poles
    assert verdict.causal_metrics is not None  # ranked against the dictionary's other latents
    assert verdict.causal_metrics.null_neurons == SMALL.causal_null_neurons
    assert AuditRecord.model_validate_json(record.model_dump_json()) == record
    report = reproduce_record(record, runner)
    assert report.ok, report.mismatches

    prompt = ReviewerAgent.probe_prompt(sub)
    assert prompt.startswith("Dictionary feature: tiny-gpt2 L1/F5 (tiny-tc)")
    task = ProposerAgent.proposal_prompt(FEATURE, scan.examples, dictionary_profile=scan.profile)
    assert "sparse transcoder (48 latents)" in task and "not a native neuron" in task


def test_feature_pages_parquet_and_follow_ups(runner, scan, tmp_path):
    record = _record(runner, scan)
    wiki = tmp_path / "wiki"
    page = write_experiment_page(record, wiki)
    assert page.parent == wiki / "experiments" / "tiny-gpt2" / "L1" / "tiny-tc" / "F5"
    text = page.read_text()
    assert "# tiny-gpt2 · L1/F5 (tiny-tc) ·" in text and "learned-basis baseline" in text
    assert "[wiki home](../../../../../README.md)" in text

    report = compile_site([record], tmp_path / "dist")
    html = (tmp_path / "dist" / "models" / "tiny-gpt2" / "L1" / "tiny-tc" / "F5.html").read_text()
    assert "tiny-tc latent" in html and "Explained variance" in html
    assert 'href="../../../../index.html"' in html  # one directory deeper than a neuron
    (row,) = pq.read_table(report.parquet).to_pylist()
    assert (row["kind"], row["dictionary"], row["feature_idx"], row["neuron_idx"]) == (
        "dictionary_feature",
        "tiny-tc",
        5,
        None,
    )
    assert row["dictionary_explained_variance"] == scan.profile.explained_variance
    home = (tmp_path / "dist" / "index.html").read_text()
    assert "latents of sparse dictionaries" in home  # counted apart from native units

    plan = plan_next(wiki, epsilon=0.0, seed=0)
    assert "opposite_pole" not in {c.kind for c in plan.candidates}
    audits = [cmd for c in plan.candidates for cmd in c.commands if "openinterp audit" in cmd]
    assert audits and all("--dictionary tiny-tc --feature 5" in cmd for cmd in audits)
    assert not any("--pole" in cmd for cmd in audits)


def test_cli_samples_a_random_latent_of_the_dictionary(runner, spec, tmp_path, monkeypatch):
    loads = []

    def load(model_name, device, dtype, revision=None, attn=None):
        loads.append((model_name, attn))
        return runner

    monkeypatch.setattr("openinterp.cli._load_runner", load)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    args = ["audit", "--dictionary", "tiny-tc", "--random", "--seed", "3",
            "--corpus", "corpus.txt", "--evidence-out", "ev.json"]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    evidence = ScanEvidence.model_validate_json((tmp_path / "ev.json").read_text())
    ident = evidence.identifier
    assert ident.kind == "dictionary_feature" and ident.layer in spec.layers
    assert ident.feature_idx < N_FEATURES and evidence.provenance.selection == "random"
    assert evidence.dictionary_profile.path == f"tc{ident.layer}/sae.safetensors"
    assert evidence.vocabulary_lens is not None and evidence.input_attribution is None
    assert loads == [("tiny-gpt2", None)]  # the dictionary names the model
    assert "Dictionary feature:" in (tmp_path / "ev.prompt.md").read_text()

    for bad_args, message in [
        (["--dictionary", "tiny-tc", "--model", "other"], "a dictionary for tiny-gpt2"),
        (["--dictionary", "nope", "--layer", "0", "--feature", "1"], "unknown dictionary"),
        (["--feature", "1", "--pole", "positive"], "a feature has neither"),
        (["--unit", "feature", "--random"], "needs --dictionary"),
    ]:
        bad = CliRunner().invoke(cli, ["audit", *bad_args])
        assert bad.exit_code != 0 and message in bad.output, bad.output
