"""Auditing a sparse-dictionary latent end to end, on a randomly initialised GPT-2 and a
random transcoder stored locally (no download)."""

import json
from dataclasses import replace

import numpy as np
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
from openinterp.core.schemas import (
    AuditRecord,
    DictionaryProfile,
    FeatureIdentifier,
    ScanEvidence,
)
from openinterp.harness import dictionaries
from openinterp.harness.dictionaries import DictionarySpec, load_spec
from openinterp.harness.features import patch_features, scan_feature
from openinterp.harness.hooks import NeuronHookRunner, activation_histogram
from openinterp.harness.patchscope import write_direction
from openinterp.harness.sites import dictionary_io
from openinterp.harness.steering import steering_interventions
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


def test_dictionary_sites_read_the_residual_the_mlp_input_its_normalisation_and_output(runner):
    seen = _captured(runner, TEXTS[0])
    pre = seen["pre"]
    variance = pre.var(-1, correction=0, keepdim=True)
    standardised = (pre - pre.mean(-1, keepdim=True)) / (variance + 1e-5).sqrt()
    for site, expected in [
        ("resid_mid", pre),
        ("mlp_in", seen["in"]),
        ("mlp_in_normalized", standardised),
        ("mlp_out", seen["out"]),
    ]:
        x, out = dictionary_io(runner, TEXTS[0], 1, site)
        torch.testing.assert_close(x.float(), expected.float(), atol=1e-5, rtol=0)
        torch.testing.assert_close(out, seen["out"])


def test_scan_measures_the_latent_and_the_dictionarys_fit(runner, spec, scan):
    unit = FeatureUnit(FEATURE, scan.profile)
    for ex in scan.examples:
        measured = unit.strengths(runner, ex.text)[0]
        torch.testing.assert_close(measured, torch.tensor(ex.activations), atol=1e-4, rtol=0)
    assert scan.examples[0].max_activation == pytest.approx(max(scan.values))

    dictionary = load_spec(spec, 1)
    xs, outs = zip(*(dictionary_io(runner, t, 1, spec.input_site) for t in TEXTS), strict=True)
    x, out = torch.cat([v[0, 1:] for v in xs]), torch.cat([v[0, 1:] for v in outs])
    acts = dictionary.encode(x)  # the fit leaves out each text's first token
    error = ((dictionary.decode(acts) - out) ** 2).sum()
    profile = scan.profile
    assert profile.explained_variance == pytest.approx(
        float(1 - error / ((out - out.mean(0)) ** 2).sum()), rel=1e-5
    )
    assert profile.mean_l0 == pytest.approx(float((acts > 0).sum(-1).float().mean()), abs=1e-3)
    every = dictionary.encode(torch.cat([v[0] for v in xs]))  # density counts every token
    assert profile.density == pytest.approx(float((every[:, 5] > 0).float().mean()), abs=1e-5)
    assert profile.dead_share == pytest.approx(float((acts.amax(0) == 0).float().mean()))
    assert (profile.n_features, profile.tokens, profile.path) == (
        N_FEATURES,
        len(scan.values),
        "tc1/sae.safetensors",
    )


def test_npz_dictionaries_by_layer_with_a_skip_connection(runner, tmp_path):
    torch.manual_seed(2)
    weights = {
        "W_enc": torch.randn(32, N_FEATURES) / 32**0.5,
        "b_enc": torch.zeros(N_FEATURES),
        "W_dec": torch.randn(N_FEATURES, 32) * 0.1,
        "b_dec": torch.zeros(32),
        "threshold": torch.full((N_FEATURES,), 0.1),
        "W_skip": torch.randn(32, 32) * 0.1,  # [d_out, d_in]
    }
    (tmp_path / "l1").mkdir()
    np.savez(tmp_path / "l1" / "params.npz", **{k: v.numpy() for k, v in weights.items()})
    spec = DictionarySpec(
        model_name="tiny-gpt2",
        kind="transcoder",
        repo=str(tmp_path),
        revision="local",
        path={1: "l1/params.npz"},
        layers=(1,),
        input_site="resid_mid",
        activation="jumprelu",
    )
    assert spec.file(1) == "l1/params.npz"
    with pytest.raises(ValueError, match="no layer 0"):
        spec.file(0)

    dictionary = load_spec(spec, 1)
    x, out = dictionary_io(runner, TEXTS[0], 1, "resid_mid")
    x, out = x[0], out[0]
    pre = x @ weights["W_enc"] + weights["b_enc"]
    acts = pre * (pre > 0.1)
    expected = acts @ weights["W_dec"] + x @ weights["W_skip"].T
    torch.testing.assert_close(dictionary.reconstruct(x), expected, atol=1e-5, rtol=0)

    scan = scan_feature(runner, [TEXTS[0]], FEATURE, spec, k=1)  # the fit includes the skip
    expected, out = expected[1:], out[1:]  # and leaves out the first token
    error = ((expected - out) ** 2).sum()
    assert scan.profile.explained_variance == pytest.approx(
        float(1 - error / ((out - out.mean(0)) ** 2).sum()), rel=1e-4
    )


def test_an_lm_sae_checkpoint_reads_as_neuronpedia_reads_it(runner, tmp_path):
    torch.manual_seed(3)
    d, n, theta = 32, N_FEATURES, 0.02
    w = {
        "encoder.weight": torch.randn(n, d) / d**0.5,  # [n, d_in]
        "encoder.bias": torch.randn(n) * 0.1,
        "decoder.weight": torch.randn(d, n) * 0.3,  # [d_out, n], rows not unit-norm
        "decoder.bias": torch.randn(d) * 0.1,
    }
    norms = {"in": 0.2, "out": 0.1}  # the tiny model's residual norm is about 0.17
    folder = tmp_path / "L1R" / "checkpoints"
    folder.mkdir(parents=True)
    save_file(w, folder / "final.safetensors")
    (tmp_path / "L1R" / "hyperparams.json").write_text(
        json.dumps({"jump_relu_threshold": theta, "dataset_average_activation_norm": norms})
    )
    spec = DictionarySpec(
        model_name="tiny-gpt2",
        kind="sae",
        repo=str(tmp_path),
        revision="local",
        path="L{layer}R/checkpoints/final.safetensors",
        layers=(1,),
        input_site="resid_post",
        activation="jumprelu",
    )
    dictionary = load_spec(spec, 1)
    x, _ = dictionary_io(runner, TEXTS[0], 1, "resid_post")
    x = x[0]

    s_in, s_out = d**0.5 / norms["in"], d**0.5 / norms["out"]
    h = (x * s_in) @ w["encoder.weight"].T + w["encoder.bias"]
    active = h > theta * s_in  # SAELens's threshold
    assert 0 < int(active.sum()) < active.numel()
    dec_norm = w["decoder.weight"].norm(dim=0)
    expected = torch.where(active, h * dec_norm / s_in, torch.zeros_like(h))
    acts = dictionary.encode(x)
    torch.testing.assert_close(acts.clamp(min=0), expected, atol=1e-5, rtol=1e-5)
    reconstruction = (torch.where(active, h, 0) @ w["decoder.weight"].T + w["decoder.bias"]) / s_out
    torch.testing.assert_close(dictionary.reconstruct(x), reconstruction, atol=1e-5, rtol=1e-5)


@pytest.fixture
def gemma2_runner():
    """A random Gemma 2, whose MLP output is normalised before it joins the residual."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    return NeuronHookRunner(Gemma2ForCausalLM(config).eval(), FakeTokenizer())


def test_a_dictionary_reconstructs_and_writes_after_the_post_mlp_norm(gemma2_runner, spec):
    runner = gemma2_runner
    block = runner.model.model.layers[1]
    seen = {}
    hooks = [
        block.pre_feedforward_layernorm.register_forward_pre_hook(
            lambda _m, a: seen.__setitem__("pre", a[0])
        ),
        block.post_feedforward_layernorm.register_forward_hook(
            lambda _m, _a, o: seen.__setitem__("out", o)
        ),
    ]
    with torch.no_grad():
        runner.forward(runner.encode(TEXTS[1]))
    for h in hooks:
        h.remove()
    pre, written = seen["pre"].float(), seen["out"]
    x, out = dictionary_io(runner, TEXTS[1], 1, "mlp_in_normalized")
    torch.testing.assert_close(x, pre * (pre.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt())
    torch.testing.assert_close(out, written)
    torch.testing.assert_close(dictionary_io(runner, TEXTS[1], 1, "mlp_out")[0], written)

    profile = scan_feature(runner, TEXTS, FEATURE, spec, k=2).profile
    unit = FeatureUnit(FEATURE, profile)
    dictionary = unit.dictionary(runner)
    after = []
    with runner.hooks(**unit.ablation(runner)):  # registered after the ablation, so it sees it
        handle = block.post_feedforward_layernorm.register_forward_hook(
            lambda _m, _a, o: after.append(o)
        )
        runner.forward(runner.encode(TEXTS[1]))
    handle.remove()
    write = dictionary.feature(x, 5).unsqueeze(-1) * dictionary.w_dec[5]
    assert float(write.abs().max()) > 0
    torch.testing.assert_close(after[-1], written - write, atol=1e-6, rtol=0)


@pytest.fixture
def residual_spec(spec, monkeypatch):
    residual = replace(spec, kind="sae", input_site="resid_post")
    monkeypatch.setitem(dictionaries.DICTIONARIES, "tiny-tc", residual)
    return residual


def test_a_residual_sae_reads_and_writes_the_residual_after_its_layer(runner, residual_spec):
    block = runner.model.transformer.h[1]
    seen = []
    handle = block.register_forward_hook(lambda _m, _a, o: seen.append(o))
    with torch.no_grad():
        runner.forward(runner.encode(TEXTS[1]))
    handle.remove()
    residual = seen[-1][0] if isinstance(seen[-1], tuple) else seen[-1]
    x, out = dictionary_io(runner, TEXTS[1], 1, "resid_post")
    torch.testing.assert_close(x, residual)
    torch.testing.assert_close(out, residual)

    profile = scan_feature(runner, TEXTS, FEATURE, residual_spec, k=2).profile
    assert (profile.input_site, profile.output_site) == ("resid_post", "resid_post")
    with pytest.raises(ValueError, match="writes to resid_post, not mlp_out"):
        DictionaryProfile.model_validate({**profile.model_dump(), "output_site": "mlp_out"})

    unit = FeatureUnit(FEATURE, profile)
    dictionary = unit.dictionary(runner)
    after = []
    with runner.hooks(**unit.ablation(runner)):  # registered after the ablation, so it sees it
        handle = block.register_forward_hook(lambda _m, _a, o: after.append(o))
        runner.forward(runner.encode(TEXTS[1]))
    handle.remove()
    ablated = after[-1][0] if isinstance(after[-1], tuple) else after[-1]
    write = dictionary.feature(x, 5).unsqueeze(-1) * dictionary.w_dec[5]
    assert float(write.abs().max()) > 0
    torch.testing.assert_close(ablated, residual - write, atol=1e-6, rtol=0)


def test_a_residual_latent_is_audited_and_reproduced(runner, residual_spec, tmp_path):
    scan = scan_feature(runner, TEXTS, FEATURE, residual_spec, k=4)
    record = _record(runner, scan)
    assert record.submission.dictionary_profile.output_site == "resid_post"
    assert record.verdict.causal_metrics is not None
    report = reproduce_record(record, runner)
    assert report.ok, report.mismatches

    task = ProposerAgent.proposal_prompt(FEATURE, scan.examples, dictionary_profile=scan.profile)
    assert "decoder vector to the residual stream after this layer" in task
    page = write_experiment_page(record, tmp_path / "wiki").read_text()
    assert "of the residual stream's variance" in page


def test_ablation_removes_exactly_the_latents_write(runner, scan):
    unit = FeatureUnit(FEATURE, scan.profile)
    dictionary = unit.dictionary(runner)
    x, clean = dictionary_io(runner, TEXTS[1], 1, "mlp_in_normalized")
    seen = []  # on the MLP block, so it sees the down projection's output after the ablation
    mlp = runner.model.transformer.h[1].mlp
    handle = mlp.register_forward_hook(lambda _m, _a, out: seen.append(out))
    runner.run_with_interventions(TEXTS[1], **unit.ablation(runner))
    handle.remove()
    write = dictionary.feature(x, 5).unsqueeze(-1) * dictionary.w_dec[5]
    assert float(write.abs().max()) > 0
    torch.testing.assert_close(seen[-1], clean - write, atol=1e-6, rtol=0)


def test_steering_sets_the_latent_and_ablation_removes_its_write(runner, scan):
    unit = FeatureUnit(FEATURE, scan.profile)
    dictionary = unit.dictionary(runner)
    x, clean = dictionary_io(runner, [TEXTS[1]] * 3, 1, "mlp_in_normalized")
    seen = []  # on the MLP block, so it sees the down projection's output after steering
    mlp = runner.model.transformer.h[1].mlp
    handle = mlp.register_forward_hook(lambda _m, _a, out: seen.append(out))
    ablate, clamp = torch.tensor([False, True, False]), torch.tensor([float("nan")] * 2 + [2.0])
    with runner.hooks(**steering_interventions(runner, unit, ablate, clamp)):
        runner.forward(runner.encode([TEXTS[1]] * 3))
    handle.remove()
    a, w = dictionary.feature(x, 5).unsqueeze(-1), dictionary.w_dec[5]
    expected = torch.stack([clean[0], clean[1] - a[1] * w, clean[2] + (2.0 - a[2]) * w])
    torch.testing.assert_close(seen[-1], expected, atol=1e-6, rtol=0)


def test_patching_a_latent_moves_the_output_by_its_write_change(runner, spec, scan):
    dictionary = load_spec(spec, 1)
    source, target = "abcabc", "abcxyz"
    site = spec.input_site
    a = {t: dictionary.encode(dictionary_io(runner, t, 1, site)[0][0]) for t in (source, target)}
    _, clean = dictionary_io(runner, target, 1, site)

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
    unit = FeatureUnit(FEATURE, scan.profile)  # a Patchscope reads the decoder row
    assert torch.equal(write_direction(runner, unit), unit.dictionary(runner).w_dec[5].float())

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
    args = ["audit", "--dictionary", "tiny-tc", "--random", "--seed", "8",
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

    short = replace(spec, context_tokens=8)
    monkeypatch.setitem(dictionaries.DICTIONARIES, "tiny-short", short)
    for bad_args, message in [
        (["--dictionary", "tiny-tc", "--model", "other"], "a dictionary for tiny-gpt2"),
        (["--dictionary", "tiny-short", "--random"], "--window-tokens 8"),
        (["--dictionary", "tiny-short", "--random", "--window-tokens", "16"], "up to 8 tokens"),
        (["--dictionary", "nope", "--layer", "0", "--feature", "1"], "unknown dictionary"),
        (["--feature", "1", "--pole", "positive"], "a feature has neither"),
        (["--unit", "feature", "--random"], "needs --dictionary"),
    ]:
        bad = CliRunner().invoke(cli, ["audit", *bad_args])
        assert bad.exit_code != 0 and message in bad.output, bad.output


def test_ci_reproduces_only_preset_dictionaries(runner, scan, tmp_path, monkeypatch):
    """The model allow-list also bounds what CI downloads: a feature record must pin its
    preset's weights."""
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: runner)
    record = _record(runner, scan)
    elsewhere = scan.profile.model_copy(update={"repo": "someone/else"})
    moved = record.model_copy(
        update={
            "submission": record.submission.model_copy(update={"dictionary_profile": elsewhere})
        }
    )
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(record.model_dump_json())
    bad.write_text(moved.model_dump_json())
    args = ["reproduce", str(good), str(bad), "--allow-model", "tiny-gpt2"]
    result = CliRunner().invoke(cli, args)
    assert f"REPRODUCED: {good}" in result.output, result.output
    assert f"FAILED: {bad}" in result.output and "not a preset's" in result.output


def test_feature_scans_take_label_sets(runner, spec):
    from openinterp.harness.annotations import DocumentLabels

    langs = ["en", "fr"] * 3
    tagged = scan_feature(
        runner, TEXTS, FEATURE, spec, k=4, taggers=[DocumentLabels("language", langs)]
    )
    language = tagged.confounds.labels["language"]
    assert language.labelled_tokens == tagged.profile.tokens
    assert {s.label for s in language.levels} == {"en", "fr"}
    for ex in tagged.examples:
        assert ex.peak_labels == {"language": langs[TEXTS.index(ex.text)]}
