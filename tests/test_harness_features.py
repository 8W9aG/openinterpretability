"""The checks a latent scan and a latent patch make on their inputs (harness/features.py)."""

import pytest
import torch
from safetensors.torch import save_file

from openinterpretability.core.schemas import FeatureIdentifier
from openinterpretability.harness.dictionaries import DictionarySpec, load_spec
from openinterpretability.harness.features import patch_features, scan_feature, write_changes
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer

N_FEATURES = 16
SITE = "mlp_in_normalized"


def _ident(feature_idx=3):
    return FeatureIdentifier(
        model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=feature_idx
    )


@pytest.fixture
def spec(tmp_path):
    torch.manual_seed(1)
    weights = {
        "W_enc": torch.randn(N_FEATURES, 32) / 32**0.5,
        "b_enc": torch.zeros(N_FEATURES),
        "W_dec": torch.randn(N_FEATURES, 32) * 0.1,
        "b_dec": torch.zeros(32),
    }
    (tmp_path / "tc1").mkdir()
    save_file(weights, tmp_path / "tc1" / "sae.safetensors")
    return DictionarySpec(
        model_name="tiny-gpt2",
        kind="transcoder",
        repo=str(tmp_path),
        revision="local",
        path="tc{layer}/sae.safetensors",
        layers=(1,),
        input_site=SITE,
    )


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_latent_scan_needs_a_tokenizer(tiny_gpt2, spec):
    with pytest.raises(ValueError, match="requires a tokenizer"):
        scan_feature(NeuronHookRunner(tiny_gpt2), ["abc"], _ident(), spec)


def test_a_latent_past_the_dictionarys_width_is_refused(runner, spec):
    with pytest.raises(ValueError, match=f"the dictionary has {N_FEATURES} latents"):
        scan_feature(runner, ["abc"], _ident(N_FEATURES), spec)


def test_an_empty_corpus_has_nothing_to_scan(runner, spec):
    with pytest.raises(ValueError, match="no tokens to scan"):
        scan_feature(runner, [], _ident(), spec)


def _site_input(runner, text):
    """The dictionary's input at layer 1 on ``text``: ``[seq, d_model]``."""
    from openinterpretability.harness.sites import dictionary_io

    return dictionary_io(runner, text, 1, SITE)[0][0]


def test_prompts_of_different_lengths_compare_their_last_tokens(runner, spec):
    dictionary = load_spec(spec, 1)
    positions, index, change = write_changes(runner, dictionary, _ident(), SITE, "abcd", "ab")
    assert (positions, index) == ("last", -1)
    assert change.shape == (N_FEATURES,)
    a = {t: dictionary.encode(_site_input(runner, t))[-1] for t in ("abcd", "ab")}
    torch.testing.assert_close(change[3], (a["abcd"][3] - a["ab"][3]).abs())


def _no_effect(_patched, _target):
    return 0.0


def test_patching_takes_one_target_prompt(runner, spec):
    dictionary = load_spec(spec, 1)
    with pytest.raises(ValueError, match="one source and one target prompt"):
        patch_features(runner, dictionary, SITE, "ab", ["ab", "cd"], 1, [3], effect=_no_effect)


def test_patching_every_position_needs_prompts_of_one_length(runner, spec):
    dictionary = load_spec(spec, 1)
    with pytest.raises(ValueError, match="differ in length; use positions='last'"):
        patch_features(runner, dictionary, SITE, "abc", "ab", 1, [3], effect=_no_effect)
