"""Loading dictionary weights: a Hub download, and files that lack tensors their reading needs."""

from pathlib import Path

import huggingface_hub
import pytest
import torch
from safetensors.torch import save_file

from openinterpretability.core.profiles import DictionaryProfile
from openinterpretability.harness import dictionaries
from openinterpretability.harness.dictionaries import load_dictionary

D, N = 4, 3


@pytest.fixture(autouse=True)
def fresh_cache():
    dictionaries._load.cache_clear()
    yield
    dictionaries._load.cache_clear()


def _save(folder: Path) -> str:
    g = torch.Generator().manual_seed(0)
    weights = {"W_enc": torch.randn(D, N, generator=g), "b_enc": torch.zeros(N),
               "W_dec": torch.randn(N, D, generator=g), "b_dec": torch.zeros(D)}  # fmt: skip
    save_file(weights, folder / "sae.safetensors")
    return "sae.safetensors"


def _profile(repo, path, activation, kind="sae"):
    return DictionaryProfile(
        kind=kind, repo=str(repo), path=path, input_site="mlp_out", activation=activation,
        n_features=N, density=0.5, tokens=1,
    )  # fmt: skip


def test_a_file_not_on_disk_is_downloaded_from_the_hub(tmp_path, monkeypatch):
    calls = []

    def download(repo, path, revision=None):
        calls.append((repo, path, revision))
        return str(tmp_path / "cached" / path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    found = dictionaries._fetch("org/sae", "abc123", "layer_3/sae.safetensors")
    assert found == tmp_path / "cached" / "layer_3" / "sae.safetensors"
    assert calls == [("org/sae", "layer_3/sae.safetensors", "abc123")]


def test_a_jumprelu_dictionary_needs_a_threshold(tmp_path):
    path = _save(tmp_path)
    with pytest.raises(ValueError, match="JumpReLU dictionary without a threshold tensor"):
        load_dictionary(_profile(tmp_path, path, "jumprelu"))


def test_a_causal_importance_dictionary_needs_a_gate(tmp_path):
    path = _save(tmp_path)
    with pytest.raises(ValueError, match="has neither gate"):
        load_dictionary(_profile(tmp_path, path, "causal_importance", kind="spd"))
