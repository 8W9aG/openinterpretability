"""A `basis` over the MLP's hidden activations (the rotated-basis null): with the identity
basis, its latents are exactly the native positive poles."""

import pytest
import torch
from safetensors.torch import save_file

from openinterp.agents.proposer import format_dictionary_profile
from openinterp.core.schemas import DictionaryProfile, FeatureIdentifier, NeuronIdentifier
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.units import FeatureUnit, NeuronUnit
from tests.conftest import FakeTokenizer

LAYER, INDEX, TEXT = 1, 7, "hello hello quoted 'text'"


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def _identity_profile(runner, tmp_path) -> DictionaryProfile:
    w_down = runner.down_proj(LAYER).weight.detach().float()  # GPT-2 Conv1D: [d_mlp, d_model]
    width = w_down.shape[0]
    save_file(
        {
            "W_enc": torch.eye(width),
            "b_enc": torch.zeros(width),
            "W_dec": w_down.contiguous(),
            "b_dec": torch.zeros(w_down.shape[1]),
        },
        str(tmp_path / "layer.safetensors"),
    )
    return DictionaryProfile(
        kind="basis",
        repo=str(tmp_path),
        path="layer.safetensors",
        input_site="mlp_hidden",
        activation="relu",
        n_features=width,
        density=0.5,
        tokens=1,
    )


def test_an_identity_basis_latent_is_the_native_positive_pole(runner, tmp_path):
    profile = _identity_profile(runner, tmp_path)
    ident = FeatureIdentifier(
        model_name="tiny-gpt2", layer=LAYER, dictionary="b", feature_idx=INDEX
    )
    latent = FeatureUnit(ident, profile)
    pole = NeuronUnit(
        NeuronIdentifier(model_name="tiny-gpt2", layer=LAYER, neuron_idx=INDEX, pole="positive")
    )
    torch.testing.assert_close(latent.strengths(runner, TEXT), pole.strengths(runner, TEXT))
    inputs = runner.encode(TEXT)
    ablated_latent = runner.run_with_interventions(inputs, **latent.ablation(runner))
    ablated_pole = runner.run_with_interventions(inputs, **pole.ablation(runner))
    torch.testing.assert_close(ablated_latent, ablated_pole, atol=1e-5, rtol=1e-4)


def test_the_prompt_names_neither_arm_of_a_basis(runner, tmp_path):
    text = format_dictionary_profile(_identity_profile(runner, tmp_path))
    assert "MLP hidden space" in text
    assert not any(word in text.lower() for word in ("identity", "rotat"))
