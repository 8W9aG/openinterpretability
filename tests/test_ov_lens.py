"""A head's OV circuit read through the vocabulary (harness/ov_lens.py, #5062)."""

import pytest
import torch

from openinterpretability.agents.proposer_profiles import format_head_profile
from openinterpretability.harness import ov_lens as ov
from openinterpretability.harness.heads import scan_head
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer
from tests.test_head_audit import HEAD


@pytest.fixture
def runner(tiny_gpt2):
    tiny_gpt2.set_attn_implementation("eager")
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_scan_reads_the_heads_ov_circuit_and_maps_its_attended_tokens(runner):
    scan = scan_head(runner, ["abcabc defdef", "ghi ghi jkl"], HEAD, k=3)
    lens = scan.profile.ov_lens
    assert -1.0 <= lens.copying_score <= 1.0 and lens.tokens > 0 and lens.median_rank >= 1
    assert lens.mappings and all(len(m.top) == 5 for m in lens.mappings)
    note = format_head_profile(scan.profile)
    assert "copying score" in note and "attended tokens map to" in note


def test_a_model_whose_attention_weights_are_unknown_gets_no_lens(runner, monkeypatch):
    monkeypatch.setattr(ov, "attention_reads", lambda runner, layer: None)
    assert scan_head(runner, ["abcabc defdef"], HEAD, k=2).profile.ov_lens is None
    with pytest.raises(ValueError, match="aren't supported"):
        ov.ov_circuit(runner, 0, 0)


def test_the_unembedding_carries_the_final_layernorm(runner):
    unembed = ov.unembedding(runner)
    assert torch.allclose(unembed.mean(-1), torch.zeros(unembed.shape[0]), atol=1e-5)


def test_a_circuit_with_no_eigenvalues_scores_zero():
    assert ov.copying_score(torch.eye(2), torch.zeros(2, 2), torch.eye(2)) == 0.0


def test_a_profile_without_a_lens_adds_no_note(runner):
    profile = scan_head(runner, ["abcabc defdef"], HEAD, k=2).profile
    assert "OV circuit" not in format_head_profile(profile.model_copy(update={"ov_lens": None}))


def test_with_no_final_norm_the_unembedding_is_the_output_weights(runner, monkeypatch):
    monkeypatch.setattr(runner, "final_norm", lambda: None)
    weights = runner.model.get_output_embeddings().weight.detach().float()
    assert torch.equal(ov.unembedding(runner), weights)


def test_a_norm_with_a_gain_but_no_centring_only_scales(runner, monkeypatch):
    norm = torch.nn.RMSNorm(runner.model.config.n_embd)
    monkeypatch.setattr(runner, "final_norm", lambda: norm)
    weights = runner.model.get_output_embeddings().weight.detach().float()
    assert torch.allclose(ov.unembedding(runner), weights * norm.weight.detach())
